import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_processing import get_dataset
from utils import cosine_sim, get_text_embedding

import pandas as pd
import numpy as np
import gensim.downloader as api


def test_get_dataset():
    df = get_dataset(output=False, visualize=False)
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (40000, 10)


def test_add_features():
    data = {
        "Text": ["Very good thing", "Bad thing, I want die cause it!"],
        "Summary": ["Good", "Bad"]
    }
    df = pd.DataFrame(data)

    word_vectors = api.load("glove-wiki-gigaword-50")
    good_emb = word_vectors["good"]
    bad_emb = word_vectors["bad"]

    good_sim_1 = cosine_sim(get_text_embedding(df["Text"][0], word_vectors), good_emb)
    bad_sim_1 = cosine_sim(get_text_embedding(df["Text"][0], word_vectors), bad_emb)

    good_sim_2 = cosine_sim(get_text_embedding(df["Text"][1], word_vectors), good_emb)
    bad_sim_2 = cosine_sim(get_text_embedding(df["Text"][1], word_vectors), bad_emb)

    # Check if first phraze is near to good than bad
    assert good_sim_1 > bad_sim_1  

    # Check if first phraze is near to bad than good
    assert bad_sim_2 > good_sim_2  
