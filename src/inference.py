'''
Example of launch of model with 2 params:
python src/inference.py ./runs/train1/best_catboost_model.cbm  "Very good!" "I really like that masterpiece!"
 
Example of request sent using curl:
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d "{\"summary\": \"Great product!\", \"text\": \"This product works perfectly and I love it.\", \"HelpfulnessNumerator\": 5, \"HelpfulnessDenominator\": 7}"
'''

import os
import sys
import logging
import numpy as np
from catboost import CatBoostRegressor
from utils import get_text_embedding, cosine_sim
import gensim.downloader as api

class InferenceEngine:
    """Handles model loading and prediction operations"""
    
    def __init__(self, model_path=None, word_vectors=None):
        """Initialize with optional model path and word vectors"""
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.word_vectors = word_vectors
        
        if model_path:
            self.load_model(model_path)
        
        if not self.word_vectors:
            self.load_word_vectors()

    def load_model(self, model_path):
        """Load CatBoost model from file"""
        try:
            if not os.path.exists(model_path):
                self.logger.error(f"Model not found at path: {model_path}")
                raise FileNotFoundError(f"Model not found: {model_path}")
            
            self.logger.info("Loading model...")
            self.model = CatBoostRegressor()
            self.model.load_model(model_path)
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def load_word_vectors(self):
        """Load word vectors if not provided"""
        try:
            self.logger.info("Loading word vectors...")
            self.word_vectors = api.load("glove-wiki-gigaword-50")
            self.logger.info("Word vectors loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load word vectors: {str(e)}")
            raise

    def predict(self, summary, text, HelpfulnessNumerator=1, HelpfulnessDenominator=1, verbose=False):
        """Make prediction on input text"""
        try:
            if verbose:
                self.logger.info("Calculating cosine similarities...")
            
            good_emb = self.word_vectors["good"]
            bad_emb = self.word_vectors["bad"]
            
            cos_sim_good_text = cosine_sim(get_text_embedding(text, self.word_vectors), good_emb)
            cos_sim_bad_text = cosine_sim(get_text_embedding(text, self.word_vectors), bad_emb)
            cos_sim_good_summary = cosine_sim(get_text_embedding(summary, self.word_vectors), good_emb)
            cos_sim_bad_summary = cosine_sim(get_text_embedding(summary, self.word_vectors), bad_emb)
            
            input_data = np.array([
                cos_sim_good_text,
                cos_sim_bad_text,
                cos_sim_good_summary,
                cos_sim_bad_summary,
                HelpfulnessNumerator,
                HelpfulnessDenominator
            ]).reshape(1, -1)
            
            if input_data.shape[1] != len(self.model.feature_names_):
                self.logger.error(f"Feature count mismatch. Expected: {len(self.model.feature_names_)}, Got: {input_data.shape[1]}")
                raise ValueError(f"Wrong features count {len(self.model.feature_names_)}, получено {input_data.shape[1]}")
            
            if verbose:
                self.logger.info('Running inference...')
            
            prediction = self.model.predict(input_data)
            
            if verbose:
                self.logger.info(f"Prediction result: {prediction[0]}")
            
            return prediction[0]
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

if __name__ == '__main__':
    try:
        logger = logging.getLogger(__name__)
        logger.info("Starting inference script")
        
        argv = sys.argv
        model_path = argv[1]
        input_data = argv[2:]
        summary = input_data[0]
        text = input_data[1]
        
        if len(input_data) == 4:
            HelpfulnessNumerator = input_data[2]
            HelpfulnessDenominator = input_data[3]
        else:
            HelpfulnessNumerator, HelpfulnessDenominator = 1, 1
        
        engine = InferenceEngine(model_path)
        
        logger.info(f"Running inference for text: {text[:50]}...")
        result = engine.predict(
            summary=summary,
            text=text,
            HelpfulnessNumerator=HelpfulnessNumerator,
            HelpfulnessDenominator=HelpfulnessDenominator,
            verbose=True
        )
        
        logger.info(f"Inference completed with result: {result}")
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise