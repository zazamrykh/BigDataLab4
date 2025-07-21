'''
Database Consumer for Kafka messages
This service consumes messages from Kafka and saves them to the database
'''

import os
import sys
import time
import json
import logging
import datetime
from kafka import KafkaConsumer
import hvac
import psycopg2
from psycopg2.extras import RealDictCursor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatabaseConsumer:
    """Kafka consumer for saving predictions to database"""

    def __init__(self):
        """Initialize consumer"""
        self.vault_client = None
        self.vault_connected = False
        self.db_connection = None
        self.db_connected = False
        self.kafka_consumer = None
        self.kafka_connected = False

        # Initialize Vault client
        self._setup_vault()

        # Initialize database connection
        self._setup_database()

        # Initialize Kafka consumer
        self._setup_kafka()

    def _setup_vault(self):
        """Initialize Vault client"""
        try:
            # Get Vault address from environment variable
            vault_addr = os.environ.get("VAULT_ADDR", "http://vault:8200")
            logger.info(f"Using Vault address: {vault_addr}")

            # Create Vault client
            self.vault_client = hvac.Client(url=vault_addr)

            # Use root token for testing
            self.vault_client.token = os.environ.get("VAULT_TOKEN", "root")

            # Check if client is authenticated
            if self.vault_client.is_authenticated():
                logger.info("Successfully authenticated with Vault")
                self.vault_connected = True
            else:
                logger.warning("Failed to authenticate with Vault")
                self.vault_connected = False
        except Exception as e:
            logger.error(f"Error connecting to Vault: {e}")
            self.vault_connected = False

    def _get_db_credentials(self):
        """Get database credentials from Vault or environment variables"""
        if self.vault_connected:
            try:
                # Read database credentials from Vault
                response = self.vault_client.secrets.kv.v2.read_secret_version(
                    path='database/credentials',
                    mount_point='kv'
                )

                if response and 'data' in response and 'data' in response['data']:
                    logger.info("Successfully retrieved database credentials from Vault")
                    creds = response['data']['data']
                    return {
                        'host': creds.get('host', 'postgres'),
                        'port': int(creds.get('port', '5432')),
                        'dbname': creds.get('dbname', 'reviewdb'),
                        'user': creds.get('username', 'postgres'),
                        'password': creds.get('password', 'postgres')
                    }
                else:
                    logger.warning("Failed to retrieve database credentials from Vault")
            except Exception as e:
                logger.error(f"Error retrieving database credentials from Vault: {e}")

        # Fall back to environment variables
        logger.info("Using database credentials from environment variables")
        return {
            'host': os.environ.get("DB_HOST", "postgres"),
            'port': int(os.environ.get("DB_PORT", "5432")),
            'dbname': os.environ.get("DB_NAME", "reviewdb"),
            'user': os.environ.get("DB_USER", "postgres"),
            'password': os.environ.get("DB_PASSWORD", "postgres")
        }

    def _setup_database(self):
        """Initialize database connection and create tables if needed"""
        max_retries = 5
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                # Get database credentials
                db_creds = self._get_db_credentials()
                logger.info(f"Connecting to database at {db_creds['host']}:{db_creds['port']}/{db_creds['dbname']}")

                # Connect to database
                self.db_connection = psycopg2.connect(**db_creds)
                self.db_connection.autocommit = True

                # Create tables if they don't exist
                cursor = self.db_connection.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    summary TEXT,
                    text TEXT,
                    helpfulness_numerator INTEGER,
                    helpfulness_denominator INTEGER,
                    prediction FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)

                logger.info("Successfully connected to database and created tables")
                self.db_connected = True
                return
            except Exception as e:
                logger.error(f"Failed to connect to database (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

        logger.error(f"Failed to connect to database after {max_retries} attempts")
        raise Exception("Failed to connect to database")

    def _get_kafka_credentials(self):
        """Get Kafka credentials from Vault or environment variables"""
        if self.vault_connected:
            try:
                # Read Kafka credentials from Vault
                response = self.vault_client.secrets.kv.v2.read_secret_version(
                    path='kafka/credentials',
                    mount_point='kv'
                )

                if response and 'data' in response and 'data' in response['data']:
                    logger.info("Successfully retrieved Kafka credentials from Vault")
                    return response['data']['data'].get('bootstrap_servers', 'kafka:9092')
                else:
                    logger.warning("Failed to retrieve Kafka credentials from Vault")
            except Exception as e:
                logger.error(f"Error retrieving Kafka credentials from Vault: {e}")

        # Fall back to environment variables
        logger.info("Using Kafka credentials from environment variables")
        return os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

    def _setup_kafka(self):
        """Initialize Kafka consumer"""
        max_retries = 10
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                # Get Kafka credentials
                bootstrap_servers = self._get_kafka_credentials()
                logger.info(f"Connecting to Kafka at {bootstrap_servers}")

                # Create Kafka consumer
                self.kafka_consumer = KafkaConsumer(
                    'predictions',
                    bootstrap_servers=bootstrap_servers,
                    auto_offset_reset='earliest',
                    enable_auto_commit=True,
                    group_id='database-consumer-group',
                    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
                )

                logger.info(f"Successfully connected to Kafka at {bootstrap_servers}")
                self.kafka_connected = True
                return
            except Exception as e:
                logger.error(f"Failed to connect to Kafka (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)

        logger.error(f"Failed to connect to Kafka after {max_retries} attempts")
        raise Exception("Failed to connect to Kafka")

    def save_to_database(self, prediction_data):
        """Save prediction data to database"""
        try:
            cursor = self.db_connection.cursor()

            # Insert data into the table
            cursor.execute("""
            INSERT INTO predictions (
                summary, text, helpfulness_numerator, helpfulness_denominator, prediction
            ) VALUES (
                %s, %s, %s, %s, %s
            )
            """, (
                prediction_data['summary'],
                prediction_data['text'],
                prediction_data['helpfulness_numerator'],
                prediction_data['helpfulness_denominator'],
                float(prediction_data['prediction'])
            ))

            logger.info("Prediction successfully saved to the database")
            return True
        except Exception as e:
            logger.error(f"Failed to save prediction to database: {e}")
            return False

    def run(self):
        """Run the consumer"""
        logger.info("Starting Kafka consumer...")

        if not self.kafka_connected:
            logger.error("Kafka not connected, cannot consume messages")
            return

        if not self.db_connected:
            logger.error("Database not connected, cannot save predictions")
            return

        try:
            for message in self.kafka_consumer:
                prediction_data = message.value
                logger.info(f"Received prediction: {prediction_data['prediction']}")

                # Save prediction to database
                success = self.save_to_database(prediction_data)
                if success:
                    logger.info(f"Successfully saved prediction to database: {prediction_data['prediction']}")
                else:
                    logger.error(f"Failed to save prediction to database")
        except Exception as e:
            logger.error(f"Error in Kafka consumer: {e}")
            raise

if __name__ == "__main__":
    try:
        logger.info("Starting database consumer service")
        consumer = DatabaseConsumer()
        consumer.run()
    except Exception as e:
        logger.error(f"Database consumer service failed: {str(e)}")
        raise
