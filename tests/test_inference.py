import pytest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from inference import InferenceEngine


def test_inference():
    model_path = os.path.join('runs', 'train1', 'best_catboost_model.cbm')
    if not os.path.exists(model_path):
        pytest.skip("Model file not found, skipping test")
    
    try:
        engine = InferenceEngine(model_path)
        
        summary = "Very good!"
        text = "Really very good!"
        
        prediction = engine.predict(
            summary=summary,
            text=text,
            verbose=False
        )
        
        assert isinstance(prediction, float)
    except Exception as e:
        pytest.fail(f"Inference test failed: {str(e)}")