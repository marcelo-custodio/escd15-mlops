import mlflow
import mlflow.pyfunc

# Caminho para o modelo registrado (pode ser um caminho local ou via tracking server)
MODEL_URI = "runs:/<RUN_ID>/<artifact_path>"

mlflow.set_tracking_uri("http://<your-tracking-server>:5000")

def load_model():
    model = mlflow.pyfunc.load_model(MODEL_URI)
    return model