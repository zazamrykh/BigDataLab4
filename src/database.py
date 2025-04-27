import os
import time
import psycopg2
import logging
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
from utils import load_config

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_connection(max_retries=5, retry_delay=5):
    """
    Creates a connection to the PostgreSQL database using parameters from config.ini
    and environment variables. Includes retry logic for better reliability.
    
    Args:
        max_retries: Maximum number of connection attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        Connection object
    """
    config = load_config("src/config.ini")
    db_config = config["database"] if "database" in config else {}
    
    # Get connection parameters from environment variables
    host = os.environ.get("DB_HOST", db_config.get("host", "localhost"))
    port = int(os.environ.get("DB_PORT", db_config.get("port", "5432")))
    dbname = os.environ.get("DB_NAME", db_config.get("dbname", "reviewdb"))
    user = os.environ.get("DB_USER", db_config.get("user", "postgres"))
    password = os.environ.get("DB_PASSWORD", db_config.get("password", "postgres"))
    
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
            