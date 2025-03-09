'''
Example of launch of api with 2 params:
python src/api.py "glove-wiki-gigaword-50" "./runs/train1/best_catboost_model.cbm"
Or with only path to model:
python src/api.py "./runs/train1/best_catboost_model.cbm"
'''

import os
import sys
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostRegressor
import gensim.downloader as api
from inference import run 

app = FastAPI()
word_vectors = None
model = None

class InputData(BaseModel):
    summary: str
    text: str
    HelpfulnessNumerator: int = 1
    HelpfulnessDenominator: int = 1

@app.post("/predict")
def predict(data: InputData):
    prediction = run(
        model=model,
        summary=data.summary,
        text=data.text,
        HelpfulnessNumerator=data.HelpfulnessNumerator,
        HelpfulnessDenominator=data.HelpfulnessDenominator,
        output=False,
        word_vectors=word_vectors,
    )
    return {"prediction": prediction}


if __name__ == "__main__":
    argv = sys.argv
    match(len(argv)):
        case 1:
            embdes = "glove-wiki-gigaword-50"
            model_path = './runs/train1/best_catboost_model.cbm'
        case 2:  # consider only model path is given
            embdes = "glove-wiki-gigaword-50"
            model_path = argv[1]
        case 3:
            word_vectors = argv[1]
            model_path = argv[2]

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found that way: {model_path}")
    
    word_vectors = api.load("glove-wiki-gigaword-50")
    model = CatBoostRegressor()
    model.load_model(model_path)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
