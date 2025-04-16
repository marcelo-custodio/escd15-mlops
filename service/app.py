from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
from model_utils import load_model

app = FastAPI(title="MLflow Model API")

# Carrega o modelo uma vez ao iniciar o app
model = load_model()

# Define os dados de entrada (ajuste de acordo com seu modelo)
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    # ...adicione outros campos

@app.get("/")
def root():
    return {"message": "API para servir modelo MLflow"}

@app.post("/predict")
def predict(data: List[InputData]):
    # Converte lista de objetos para DataFrame
    df = pd.DataFrame([d.dict() for d in data])
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}
