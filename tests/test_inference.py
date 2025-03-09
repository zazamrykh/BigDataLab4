import pytest
import numpy as np
import os
import sys

from catboost import CatBoostRegressor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from inference import run
import gensim.downloader as api


def test_run():
    model_path = r'C:\Projects\itmo-dir\big-data\lab1\runs\train1\best_catboost_model.cbm'
    if not os.path.exists(model_path):
        return
    
    model = CatBoostRegressor()   
    model.load_model(model_path)
    
    word_vectors = api.load("glove-wiki-gigaword-50")
    
    summary = "Very good!"
    text = "Really very good!"
    
    run(model=model, summary=summary, text=text, word_vectors=word_vectors)
    