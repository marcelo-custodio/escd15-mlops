import mlflow
import mlflow.pyfunc
from pydantic import BaseModel, create_model
from typing import Any
from parameters import model_name, mapping_types

def start_mlflow():
    uri = "sqlite:///../mlflow.db"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("customerchurn")

    return mlflow.tracking.MlflowClient(tracking_uri=uri)

def search_model(client):
    models = client.search_model_versions(
        f"name = '{model_name}'"
    )
    model = [model for model in models if model.current_stage == 'Production']
    if len(model) == 0:
        return None

    return f"models:/{model_name}/{model[0].version}"

def generate_pydantic_from_mlflow_model(model) -> BaseModel:
    inputs = model.metadata.signature.inputs
    fields = {}

    for col in inputs.inputs:
        name = col.name
        dtype = col.type
        py_type = mapping_types.get(str(dtype.name).lower(), Any)
        fields[name] = (py_type, ...)

    PydanticInputModel = create_model("ModelInput", **fields)
    return PydanticInputModel

def load_model():
    client = start_mlflow()
    model_uri = search_model(client)
    model = mlflow.pyfunc.load_model(model_uri)
    input_class = generate_pydantic_from_mlflow_model(model)
    return model, input_class