from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from model_utils import load_model

app = FastAPI(title="MLflow Model API")

# Carrega o modelo uma vez ao iniciar o app
model, InputData = load_model()

@app.get("/")
def root():
    return {"message": "API para servir modelo MLflow"}

@app.post("/predict")
def predict(data: List[InputData]):
    # Converte lista de objetos para DataFrame
    df = pd.DataFrame([d.dict() for d in data])
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}
