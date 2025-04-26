from fastapi import FastAPI
from typing import List
import pandas as pd
from ml_model.model_utils import load_model, load_pipelline, apply_pipeline
from pydantic import create_model
from ml_model.parameters import columns_to_consider, target_column
from ml_model.main import main

app = FastAPI(title="MLflow Model API")

# Carrega o modelo uma vez ao iniciar o app
class Model:
    def __init__(self):
        try:
            self.model = load_model()
            self.pipeline = load_pipelline()
        except Exception:
            self.model = None
            self.pipeline = None

    def reload(self):
        self.model = load_model()
        self.pipeline = load_pipelline()

model = Model()
input_data = {k:v for k,v in columns_to_consider.items() if k != target_column}
InputData = create_model("InputData", **input_data)

@app.get("/")
def root():
    return {"message": "API para servir modelo MLflow"}

@app.post("/train")
def train():
    score, best_model = main()
    model.reload()
    return {"best_model": best_model, "f1_score": score}

@app.post("/reload")
def reload():
    model.reload()
    return {"message": "Modelo carregado!"}

@app.post("/predict")
def predict(data: List[InputData]):
    if model.model is None or model.pipeline is None:
        main()
        model.reload()

    # Converte lista de objetos para DataFrame
    df = pd.DataFrame([apply_pipeline(model.pipeline, d.dict()) for d in data])
    predictions = model.model.predict(df)
    return {"predictions": predictions.tolist()}
