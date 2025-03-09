import pytest
import numpy as np
import os
import sys
from catboost import CatBoostRegressor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from inference import run
import gensim.downloader as api

def test_trained():
    model_path = r'C:\Projects\itmo-dir\big-data\lab1\runs\train1\best_catboost_model.cbm'
    if not os.path.exists(model_path):
        return
    
    model = CatBoostRegressor()   
    model.load_model(model_path)
    
    word_vectors = api.load("glove-wiki-gigaword-50")
    
    good_summary = "Very good!"
    good_text = "Really very good! And I like it!"
    good_score = run(model=model, summary=good_summary, text=good_text, word_vectors=word_vectors)
    
    
    bad_summary = "Worth, I want die"
    bad_text = "Bad shit"
    bad_score = run(model=model, summary=bad_summary, text=bad_text, word_vectors=word_vectors)
    
    assert bad_score < good_score, f"Inequality is not follows: {bad_score} < {good_score}"
    