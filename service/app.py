from fastapi import FastAPI
from typing import List
import pandas as pd
from model_utils import load_model, load_pipelline, apply_pipeline
from pydantic import create_model
from parameters import input_fields

app = FastAPI(title="MLflow Model API")

# Carrega o modelo uma vez ao iniciar o app
model = load_model()
pipeline = load_pipelline()

InputData = create_model("InputData", **input_fields)

@app.get("/")
def root():
    return {"message": "API para servir modelo MLflow"}

@app.post("/predict")
def predict(data: List[InputData]):
    # Converte lista de objetos para DataFrame
    df = pd.DataFrame([apply_pipeline(pipeline, d.dict()) for d in data])
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8080)