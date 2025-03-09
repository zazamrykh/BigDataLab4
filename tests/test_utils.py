import sys
import os
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from utils import cosine_sim

import numpy as np

def test_cosine_sim():
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([1, 0, 0])
    assert cosine_sim(vec1, vec2) == pytest.approx(1.0, rel=1e-5)

def test_cosine_sim_orthogonal():
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([0, 1, 0])
    assert cosine_sim(vec1, vec2) == pytest.approx(0.0, rel=1e-5)

def test_cosine_sim_negative():
    vec1 = np.array([1, 0, 0])
    vec2 = np.array([-1, 0, 0])
    assert cosine_sim(vec1, vec2) == pytest.approx(-1.0, rel=1e-5)

def test_cosine_sim_raises_error():
    with pytest.raises(ValueError):
        cosine_sim(np.array([1, 0, 0]), np.array([1, 0]))
        