import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from data_processing import DataProcessor
from utils import params, cosine_sim, get_text_embedding

import pandas as pd
import numpy as np
import gensim.downloader as api


@pytest.fixture
def data_processor():
    return DataProcessor(params)


def test_get_dataset(data_processor):
    df = data_processor.get_dataset(output=False, visualize=False)
    
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (40000, 10)


def test_add_features(data_processor):
    data = {
        "Text": ["Very good thing", "Bad thing, I want die cause it!"],
        "Summary": ["Good", "Bad"]
    }
    df = pd.DataFrame(data)
    df = data_processor.add_features(df)

    word_vectors = data_processor.word_vectors
    good_emb = word_vectors["good"]
    bad_emb = word_vectors["bad"]

    good_sim_1 = cosine_sim(get_text_embedding(data["Text"][0], word_vectors), good_emb)
    bad_sim_1 = cosine_sim(get_text_embedding(data["Text"][0], word_vectors), bad_emb)

    good_sim_2 = cosine_sim(get_text_embedding(data["Text"][1], word_vectors), good_emb)
    bad_sim_2 = cosine_sim(get_text_embedding(data["Text"][1], word_vectors), bad_emb)

    # Check if first phrase is nearer to good than bad
    assert good_sim_1 > bad_sim_1

    # Check if second phrase is nearer to bad than good
    assert bad_sim_2 > good_sim_2
