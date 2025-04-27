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
from inference import run
import database  # Import the database module

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Review Rating Prediction API",
              description="API for predicting Amazon product review ratings with Oracle database integration",
              version="2.0.0")
word_vectors = None
model = None

class InputData(BaseModel):
    summary: str
    text: str
    HelpfulnessNumerator: int = 1
    HelpfulnessDenominator: int = 1

@app.post("/predict")
def predict(data: InputData):
    try:
        # Run prediction
        prediction = run(
            model=model,
            summary=data.summary,
            text=data.text,
            HelpfulnessNumerator=data.HelpfulnessNumerator,
            HelpfulnessDenominator=data.HelpfulnessDenominator,
            output=False,
            word_vectors=word_vectors,
        )
        
        # Save prediction to Oracle database
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
        
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions")
def get_predictions(limit: int = 10):
    """
    Get the latest predictions from the database.
    """
    try:
        predictions = database.get_predictions(limit)
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Error retrieving predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    argv = sys.argv
    match(len(argv)):
        case 1:
            embdes = "glove-wiki-gigaword-50"
            model_path = './runs/train1/best_catboost_model.cbm'
        case 2:  # consider only model path is given
            embdes = "glove-wiki-gigaword-50"
            model_path = argv[1]
        case 3:
            word_vectors = argv[1]
            model_path = argv[2]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found that way: {model_path}")
    
    word_vectors = api.load("glove-wiki-gigaword-50")
    model = CatBoostRegressor()
    model.load_model(model_path)
    
    # Create database tables with multiple attempts
    db_connected = False
    max_db_connection_attempts = 5
    
    for attempt in range(max_db_connection_attempts):
        try:
            logger.info(f"Attempting to connect to database (attempt {attempt + 1}/{max_db_connection_attempts})")
            database.create_tables()
            logger.info("Database tables created successfully")
            db_connected = True
            break
        except Exception as e:
            logger.error(f"Failed to create database tables (attempt {attempt + 1}): {e}")
            if attempt < max_db_connection_attempts - 1:
                wait_time = 10  # seconds
                logger.info(f"Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
    
    if not db_connected:
        logger.warning(f"Failed to connect to database after {max_db_connection_attempts} attempts")
        logger.warning("Continuing without database support")
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
