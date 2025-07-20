import os
import time
import psycopg2
import logging
import hvac
import json
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from utils import load_config

# Load environment variables from .env file (for fallback and transition)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Vault client for retrieving secrets
vault_client = None
vault_token_path = "/vault/data/app_token.txt"

def get_vault_client():
    """
    Creates or returns a Vault client.

    Returns:
        hvac.Client: Authenticated Vault client
    """
    global vault_client

    if vault_client is not None:
        return vault_client

    # Get Vault address from environment variable
    vault_addr = os.environ.get("VAULT_ADDR", "http://vault:8200")
    logger.info(f"Using Vault address: {vault_addr}")

    try:
        # Create Vault client
        logger.info(f"Creating Vault client with URL: {vault_addr}")
        vault_client = hvac.Client(url=vault_addr)
        logger.info(f"Created Vault client with URL: {vault_addr}")

        # Use root token directly for testing
        logger.info("Setting token to 'root'")
        vault_client.token = "root"
        logger.info("Set token to 'root'")

        # Check if client is authenticated
        logger.info("Checking if client is authenticated")
        try:
            is_auth = vault_client.is_authenticated()
            logger.info(f"Vault client authenticated: {is_auth}")
        except Exception as e:
            logger.error(f"Error checking authentication: {e}")
            is_auth = False

        if is_auth:
            logger.info("Successfully authenticated with Vault")
        else:
            logger.warning("Failed to authenticate with Vault")
            vault_client = None

    except Exception as e:
        logger.error(f"Error connecting to Vault: {e}")
        import traceback
        logger.error(traceback.format_exc())
        vault_client = None

    return vault_client

def get_db_credentials_from_vault():
    """
    Retrieves database credentials from Vault.

    Returns:
        dict: Database credentials or None if retrieval failed
    """
    client = get_vault_client()

    if client is None or not client.is_authenticated():
        logger.warning("Vault client not available or not authenticated. Using environment variables.")
        return None

    try:
        # Read database credentials from Vault
        response = client.secrets.kv.v2.read_secret_version(
            path='database/credentials',
            mount_point='kv'
        )

        if response and 'data' in response and 'data' in response['data']:
            logger.info("Successfully retrieved database credentials from Vault")
            return response['data']['data']
        else:
            logger.warning("Failed to retrieve database credentials from Vault")
            return None
    except Exception as e:
        logger.error(f"Error retrieving database credentials from Vault: {e}")
        return None

def get_connection(max_retries=5, retry_delay=5):
    """
    Creates a connection to the PostgreSQL database using credentials from Vault.
    Falls back to environment variables if Vault is not available.
    Includes retry logic for better reliability.

    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retries in seconds

    Returns:
        Connection object
    """
    config = load_config("src/config.ini")
    db_config = config["database"] if "database" in config else {}

    # Try to get credentials from Vault first
    vault_credentials = get_db_credentials_from_vault()

    if vault_credentials:
        # Use credentials from Vault
        host = vault_credentials.get("host", "localhost")
        port = int(vault_credentials.get("port", "5432"))
        dbname = vault_credentials.get("dbname", "reviewdb")
        user = vault_credentials.get("username", "postgres")
        password = vault_credentials.get("password", "postgres")
        logger.info("Using database credentials from Vault")
    else:
        # Fall back to environment variables
        host = os.environ.get("DB_HOST", db_config.get("host", "localhost"))
        port = int(os.environ.get("DB_PORT", db_config.get("port", "5432")))
        dbname = os.environ.get("DB_NAME", db_config.get("dbname", "reviewdb"))
        user = os.environ.get("DB_USER", db_config.get("user", "postgres"))
        password = os.environ.get("DB_PASSWORD", db_config.get("password", "postgres"))
        logger.info("Using database credentials from environment variables")

    logger.info(f"Attempting to connect to PostgreSQL database at {host}:{port}/{dbname}")

    # Try to connect with retries
    last_exception = None
    for attempt in range(max_retries):
        try:
            connection = psycopg2.connect(
                host=host,
                port=port,
                dbname=dbname,
                user=user,
                password=password
            )
            logger.info(f"Successfully connected to PostgreSQL database on attempt {attempt + 1}")
            return connection
        except Exception as e:
            last_exception = e
            logger.warning(f"Connection attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    # If we get here, all retries failed
    logger.error(f"Failed to connect to PostgreSQL database after {max_retries} attempts. Last error: {last_exception}")
    raise last_exception

def create_tables(max_retries=3):
    """
    Creates necessary tables in the database if they don't exist yet.
    Includes retry logic for better reliability.

    Args:
        max_retries: Maximum number of attempts to create tables
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            connection = get_connection()
            cursor = connection.cursor()

            # Create table for storing predictions
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

            connection.commit()
            logger.info("Tables successfully created or already exist")
            return
        except Exception as e:
            last_exception = e
            logger.warning(f"Table creation attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying table creation...")
                time.sleep(2)
        finally:
            if 'connection' in locals() and connection:
                connection.close()

    # If we get here, all retries failed
    logger.error(f"Failed to create tables after {max_retries} attempts. Last error: {last_exception}")
    raise last_exception

def save_prediction(summary, text, helpfulness_numerator, helpfulness_denominator, prediction, max_retries=3):
    """
    Saves the prediction result to the database.
    Includes retry logic for better reliability.

    Args:
        summary: Review summary
        text: Review text
        helpfulness_numerator: Helpfulness numerator
        helpfulness_denominator: Helpfulness denominator
        prediction: Prediction value
        max_retries: Maximum number of save attempts
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            connection = get_connection()
            cursor = connection.cursor()

            # Insert data into the table
            cursor.execute("""
            INSERT INTO predictions (
                summary, text, helpfulness_numerator, helpfulness_denominator, prediction
            ) VALUES (
                %s, %s, %s, %s, %s
            )
            """, (
                summary,
                text,
                helpfulness_numerator,
                helpfulness_denominator,
                float(prediction)
            ))

            connection.commit()
            logger.info("Prediction successfully saved to the database")
            return
        except Exception as e:
            last_exception = e
            logger.warning(f"Save prediction attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying save operation...")
                time.sleep(2)
        finally:
            if 'connection' in locals() and connection:
                connection.close()

    # If we get here, all retries failed
    logger.error(f"Failed to save prediction after {max_retries} attempts. Last error: {last_exception}")
    raise last_exception

def get_predictions(limit=10, max_retries=3):
    """
    Gets the latest predictions from the database.
    Includes retry logic for better reliability.

    Args:
        limit: Maximum number of predictions to retrieve
        max_retries: Maximum number of retrieval attempts

    Returns:
        List of prediction dictionaries
    """
    last_exception = None
    for attempt in range(max_retries):
        try:
            connection = get_connection()
            # Use RealDictCursor to return results as dictionaries
            cursor = connection.cursor(cursor_factory=RealDictCursor)

            # Get data from the table
            cursor.execute("""
            SELECT id, summary, text, helpfulness_numerator, helpfulness_denominator, prediction, created_at
            FROM predictions
            ORDER BY created_at DESC
            LIMIT %s
            """, (limit,))

            # Fetch all results
            results = cursor.fetchall()

            logger.info(f"Retrieved {len(results)} predictions from the database")
            return results
        except Exception as e:
            last_exception = e
            logger.warning(f"Get predictions attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying get operation...")
                time.sleep(2)
        finally:
            if 'connection' in locals() and connection:
                connection.close()

    # If we get here, all retries failed
    logger.error(f"Failed to retrieve predictions after {max_retries} attempts. Last error: {last_exception}")
    raise last_exception
