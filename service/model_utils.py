import pickle

import mlflow
import mlflow.pyfunc
from parameters import model_name

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

def load_model():
    client = start_mlflow()
    model_uri = search_model(client)
    model = mlflow.pyfunc.load_model(model_uri)
    return model

def load_pipelline():
    with open('../pipeline.pickle', 'rb') as file:
        pipeline = pickle.load(file)
    return pipeline

def apply_pipeline(pipeline, sample):
    result = {}
    for k,v in sample.items():
        func = pipeline.get(k, None)
        if func is None:
            result[k] = v
            continue

        if func['t'] == 'constant':
            result[k] = func['v'] * v
        elif func['t'] == 'sklearn':
            result[k] = func['v'].transform([[v]]).flat[0]
        else:
            result[k] = v
    return result

if __name__ == "__main__":
    pipe = load_pipelline()
    teste = {
        'Age': 25,
        'Balance': 0.0,
        'CreditScore': 600,
        'EstimatedSalary': 1800,
        'Gender': 'female',
        'Geography': 'france',
        'HasCrCard': 1,
        'IsActiveMember': 1,
        'NumOfProducts': 2,
        'Tenure': 2
    }
    res = apply_pipeline(pipe, teste)
    print()