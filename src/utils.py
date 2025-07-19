import os
import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import configparser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Params:
    def __init__(self, exp_name='prodict_review_prediction', random_seed=1337, all_data_size=40_000, train_frac=0.5):
        self.random_seed = random_seed
        self.exp_name = exp_name
        self.all_data_size = all_data_size
        self.train_frac = train_frac  # Test and val split 50/50
        
    def __str__(self):
        return ", ".join(f"{k}: {v}" for k, v in vars(self).items())
    
params = Params()

current_train_number = 1
def create_dirs():
    global current_train_number
    try:
        if not os.path.exists('./runs'):
            logger.info("Creating runs directory")
            os.makedirs('./runs')

        while os.path.exists('./runs/train' + str(current_train_number)):
            current_train_number += 1
        
        logger.info(f"Creating training directory: train{current_train_number}")
        os.makedirs('./runs/train' + str(current_train_number))
    except Exception as e:
        logger.error(f"Error creating directories: {str(e)}")
        raise
    
    
def get_output_path():
    path = './runs/train' + str(current_train_number) + '/'
    logger.debug(f"Getting output path: {path}")
    return path

def save_params(params, save_path, min_loss=None):
    try:
        logger.info(f"Saving parameters to: {save_path}")
        with open(save_path, 'w') as f:
            f.write(str(params))
            
            if min_loss is not None:
                f.write('\nMinimal loss: ' + str(min_loss))
    except Exception as e:
        logger.error(f"Error saving parameters: {str(e)}")
        raise
                   
def get_text_embedding(text, model):
    try:
        words = text.lower().split()
        vectors = [model[word] for word in words if word in model]
        if not vectors:
            logger.debug("No vectors found for text, returning zeros")
            return np.zeros(model.vector_size)
        return np.mean(vectors, axis=0)
    except Exception as e:
        logger.error(f"Error getting text embedding: {str(e)}")
        raise


def cosine_sim(vec1, vec2):
    try:
        return cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {str(e)}")
        raise

def load_config(path="config.ini"):
    try:
        logger.info(f"Loading config from: {path}")
        config = configparser.ConfigParser()
        config.read(path)
        if not config.sections():
            logger.warning(f"Config file {path} is empty or not found")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise
