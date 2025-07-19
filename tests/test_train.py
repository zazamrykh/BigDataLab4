import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from inference import InferenceEngine


def test_trained_model_behavior():
    model_path = os.path.join('runs', 'train1', 'best_catboost_model.cbm')
    if not os.path.exists(model_path):
        pytest.skip("Model file not found, skipping test")
    
    try:
        engine = InferenceEngine(model_path)
        
        good_summary = "Very good!"
        good_text = "Really very good! And I like it!"
        good_score = engine.predict(
            summary=good_summary,
            text=good_text,
            verbose=False
        )
        
        bad_summary = "Worth, I want die"
        bad_text = "Bad shit"
        bad_score = engine.predict(
            summary=bad_summary,
            text=bad_text,
            verbose=False
        )
        
        assert bad_score < good_score, f"Model behavior incorrect: {bad_score} should be < {good_score}"
    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")