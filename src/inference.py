'''
Example of launch of model with 2 params:
python src/inference.py ./runs/train1/best_catboost_model.cbm  "Very good!" "I really like that masterpiece!"
'''

import os
import sys
import numpy as np
from catboost import CatBoostRegressor
from utils import get_text_embedding, cosine_sim
import gensim.downloader as api

def run(model=None, summary="", text="", HelpfulnessNumerator=1, HelpfulnessDenominator=1, output=False, word_vectors=None, model_path=None):
    if model is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = CatBoostRegressor()
        model.load_model(model_path)
    
    if output: print('Loading embeddings...')
    if word_vectors is None:
        word_vectors = api.load("glove-wiki-gigaword-50")
    good_emb = word_vectors["good"]
    bad_emb = word_vectors["bad"]
    
    cos_sim_good_text = cosine_sim(get_text_embedding(text, word_vectors), good_emb)
    cos_sim_bad_text = cosine_sim(get_text_embedding(text, word_vectors), bad_emb)
    cos_sim_good_summary = cosine_sim(get_text_embedding(summary, word_vectors), good_emb)
    cos_sim_bad_summary = cosine_sim(get_text_embedding(summary, word_vectors), bad_emb)
    
    input_data = np.array([cos_sim_good_text, cos_sim_bad_text, cos_sim_good_summary, cos_sim_bad_summary, 
                            HelpfulnessNumerator, HelpfulnessDenominator]).reshape(1, -1)
    
    if input_data.shape[1] != len(model.feature_names_):
        raise ValueError(f"Wrong features count {len(model.feature_names_)}, получено {input_data.shape[1]}")
    
    if output: print('Run inference...')
    prediction = model.predict(input_data)
    
    if output: print("Result prediction:", prediction[0])
    return prediction[0] 
 

if __name__ == '__main__':
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
    
    if(len(input_data) == 5):
        output = bool(input_data[5])
    else:
        output = False
        
    run(model_path, summary, text, HelpfulnessNumerator, HelpfulnessDenominator, output=True)