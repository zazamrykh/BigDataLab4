'''
Example of launch of api with 2 params:
python src/api.py "glove-wiki-gigaword-50" "./runs/train1/best_catboost_model.cbm"
Or with only path to model:
python src/api.py "./runs/train1/best_catboost_model.cbm"
'''

import os
import sys
import time
import logging
import json
import datetime
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor
import gensim.downloader as api
from inference import InferenceEngine
import database
import hvac
from kafka import KafkaProducer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReviewAPI:
    """API for review prediction service with Kafka integration and Vault secret management"""

    def __init__(self, model_path=None, word_vectors=None):
        """Initialize API with model path and word vectors"""
        self.app = FastAPI(
            title="Review Rating Prediction API",
            description="API for predicting Amazon product review ratings with Kafka and Vault integration",
            version="4.0.0"
        )
        self.model = None
        self.word_vectors = word_vectors
        self.vault_client = None
        self.vault_connected = False
        self.kafka_producer = None
        self.kafka_connected = False

        # Initialize Vault client
        self._setup_vault()

        # Initialize Kafka producer
        self._setup_kafka()

        if model_path:
            self.load_model(model_path)

        self._setup_routes()

    def _setup_vault(self):
        """Initialize Vault client"""
        try:
            # Get Vault client from database module
            self.vault_client = database.get_vault_client()

            if self.vault_client and self.vault_client.is_authenticated():
                self.vault_connected = True
                logger.info("Successfully connected to Vault")
            else:
                logger.warning("Failed to connect to Vault or not authenticated")
                self.vault_connected = False
        except Exception as e:
            logger.error(f"Error setting up Vault client: {e}")
            self.vault_connected = False

    def _get_kafka_credentials(self):
        """Get Kafka credentials from Vault or environment variables"""
        if self.vault_connected:
            try:
                # Read Kafka credentials from Vault
                kafka_creds = self.vault_client.secrets.kv.v2.read_secret_version(
                    path='kafka/credentials',
                    mount_point='kv'
                )
                if kafka_creds and 'data' in kafka_creds and 'data' in kafka_creds['data']:
                    bootstrap_servers = kafka_creds['data']['data'].get('bootstrap_servers', 'kafka:9092')
                    logger.info(f"Using Kafka credentials from Vault")
                    return bootstrap_servers
                else:
                    logger.warning("Failed to retrieve Kafka credentials from Vault, using environment variables")
            except Exception as e:
                logger.error(f"Error retrieving Kafka credentials from Vault: {e}")

        # Fall back to environment variables
        bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
        logger.warning("Using environment variables for Kafka")
        return bootstrap_servers

    def _setup_kafka(self):
        """Initialize Kafka producer with retry mechanism"""
        max_retries = 10
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                # Get Kafka credentials
                bootstrap_servers = self._get_kafka_credentials()
                logger.info(f"Connecting to Kafka at {bootstrap_servers} (attempt {attempt + 1}/{max_retries})")

                # Create Kafka producer
                self.kafka_producer = KafkaProducer(
                    bootstrap_servers=bootstrap_servers,
                    value_serializer=lambda v: json.dumps(v).encode('utf-8')
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
        self.kafka_connected = False


    def load_model(self, model_path):
        """Load prediction model"""
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model not found at path: {model_path}")
                raise FileNotFoundError(f"Model not found: {model_path}")

            logger.info("Initializing inference engine...")
            self.engine = InferenceEngine(model_path)
            self.word_vectors = self.engine.word_vectors
            self.model = self.engine.model
            logger.info("Inference engine ready")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _setup_routes(self):
        """Configure API routes"""

        class InputData(BaseModel):
            summary: str
            text: str
            HelpfulnessNumerator: int = 1
            HelpfulnessDenominator: int = 1

        @self.app.post("/predict")
        async def predict(data: InputData):
            logger.info(f"Received prediction request for text: {data.text[:50]}...")
            try:
                prediction = self.engine.predict(
                    summary=data.summary,
                    text=data.text,
                    HelpfulnessNumerator=data.HelpfulnessNumerator,
                    HelpfulnessDenominator=data.HelpfulnessDenominator,
                    verbose=True
                )

                # Send prediction to Kafka
                if self.kafka_connected:
                    try:
                        message = {
                            "summary": data.summary,
                            "text": data.text,
                            "helpfulness_numerator": data.HelpfulnessNumerator,
                            "helpfulness_denominator": data.HelpfulnessDenominator,
                            "prediction": float(prediction),
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                        self.kafka_producer.send('predictions', message)
                        logger.info(f"Prediction sent to Kafka topic 'predictions'")
                    except Exception as e:
                        logger.error(f"Failed to send prediction to Kafka: {e}")
                        # Continue even if Kafka send fails
                else:
                    logger.warning("Kafka not connected, prediction not sent")

                logger.info(f"Prediction completed: {prediction}")
                return {"prediction": prediction}
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/predictions")
        async def get_predictions(limit: int = 10):
            """Get the latest predictions from the database."""
            # This endpoint is kept for backward compatibility
            # but will return an informative message
            raise HTTPException(
                status_code=501,
                detail="This endpoint is no longer available. Predictions are now sent to Kafka and stored by the consumer service."
            )

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint to verify the API is running."""
            status = {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "kafka_connected": self.kafka_connected,
                "vault_connected": self.vault_connected
            }
            logger.info(f"Health check: {status}")
            return status

        @self.app.get("/vault-status")
        async def vault_status():
            """Check Vault status and connection."""
            if not self.vault_connected:
                raise HTTPException(status_code=503, detail="Vault not available")

            try:
                # Check if we can access Vault
                vault_status = {
                    "connected": self.vault_connected,
                    "authenticated": self.vault_client.is_authenticated() if self.vault_client else False,
                    "secrets_engine": "Available" if self.vault_client and self.vault_client.sys.list_mounted_secrets_engines() else "Not available"
                }
                logger.info(f"Vault status: {vault_status}")
                return vault_status
            except Exception as e:
                logger.error(f"Error checking Vault status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/kafka-status")
        async def kafka_status():
            """Check Kafka status and connection."""
            if not self.kafka_producer:
                raise HTTPException(status_code=503, detail="Kafka producer not initialized")

            try:
                # Perform a real check by requesting metadata from the broker
                bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

                # Try to get cluster metadata - this will fail if Kafka is not available
                cluster_metadata = self.kafka_producer.bootstrap_connected()

                # Try to send a test message to verify the connection
                test_message = {
                    "test": True,
                    "timestamp": datetime.datetime.now().isoformat()
                }

                # Send the message but don't wait for it to be delivered
                future = self.kafka_producer.send('test-topic', test_message)

                # Try to get metadata for 'test-topic'
                topic_metadata = self.kafka_producer._client.cluster.available_partitions_for_topic('test-topic')

                # If we got here, the connection is working
                kafka_status = {
                    "connected": True,
                    "bootstrap_servers": bootstrap_servers,
                    "cluster_metadata_available": cluster_metadata,
                    "topic_metadata_available": topic_metadata is not None
                }
                logger.info(f"Kafka status: {kafka_status}")
                return kafka_status
            except Exception as e:
                logger.error(f"Error checking Kafka status: {e}")
                # Update the kafka_connected flag
                self.kafka_connected = False
                raise HTTPException(status_code=503, detail=f"Kafka connection error: {str(e)}")

        @self.app.on_event("startup")
        async def startup_event():
            logger.info("API server started")

    def run(self, host="0.0.0.0", port=8000):
        """Run the API server"""
        import uvicorn
        logger.info("Starting uvicorn server...")
        uvicorn.run(self.app, host=host, port=port)

if __name__ == "__main__":
    try:
        logger.info("Starting API service")
        argv = sys.argv

        if len(argv) == 1:
            model_path = './runs/train1/best_catboost_model.cbm'
        else:
            model_path = argv[1] if len(argv) >= 2 else './runs/train1/best_catboost_model.cbm'

        api = ReviewAPI(model_path)
        api.run()
    except Exception as e:
        logger.error(f"API service failed: {str(e)}")
        raise
