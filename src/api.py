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
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor
import gensim.downloader as api
from inference import InferenceEngine
import database
import hvac

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ReviewAPI:
    """API for review prediction service with database integration and Vault secret management"""

    def __init__(self, model_path=None, word_vectors=None):
        """Initialize API with model path and word vectors"""
        self.app = FastAPI(
            title="Review Rating Prediction API",
            description="API for predicting Amazon product review ratings with Vault secret management",
            version="3.0.0"
        )
        self.model = None
        self.word_vectors = word_vectors
        self.vault_client = None
        self.vault_connected = False

        # Initialize Vault client
        self._setup_vault()

        # Initialize database connection
        self._setup_database()

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

    def _setup_database(self):
        """Initialize database connection"""
        self.db_connected = False
        max_db_connection_attempts = 5

        for attempt in range(max_db_connection_attempts):
            try:
                logger.info(f"Attempting to connect to database (attempt {attempt + 1}/{max_db_connection_attempts})")
                database.create_tables()
                logger.info("Database tables created successfully")
                self.db_connected = True
                break
            except Exception as e:
                logger.error(f"Failed to create database tables (attempt {attempt + 1}): {e}")
                if attempt < max_db_connection_attempts - 1:
                    wait_time = 10  # seconds
                    logger.info(f"Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)

        if not self.db_connected:
            logger.warning(f"Failed to connect to database after {max_db_connection_attempts} attempts")
            logger.warning("Continuing without database support")

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

                if self.db_connected:
                    try:
                        database.save_prediction(
                            summary=data.summary,
                            text=data.text,
                            helpfulness_numerator=data.HelpfulnessNumerator,
                            helpfulness_denominator=data.HelpfulnessDenominator,
                            prediction=prediction
                        )
                        logger.info(f"Prediction saved to database: {prediction}")
                    except Exception as e:
                        logger.error(f"Failed to save prediction to database: {e}")
                        # Continue even if database save fails

                logger.info(f"Prediction completed: {prediction}")
                return {"prediction": prediction}
            except Exception as e:
                logger.error(f"Prediction failed: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/predictions")
        async def get_predictions(limit: int = 10):
            """Get the latest predictions from the database."""
            if not self.db_connected:
                raise HTTPException(status_code=503, detail="Database not available")

            try:
                predictions = database.get_predictions(limit)
                logger.info(f"Retrieved {len(predictions)} predictions from database")
                return {"predictions": predictions}
            except Exception as e:
                logger.error(f"Error retrieving predictions: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health_check():
            """Health check endpoint to verify the API is running."""
            status = {
                "status": "healthy",
                "model_loaded": self.model is not None,
                "database_connected": self.db_connected,
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
