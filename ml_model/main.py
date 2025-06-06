from ml_model.preprocessing import get_dataset, split_and_clean
from ml_model.parameters import target_column, model_name
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from ml_model.model_utils import start_mlflow

def train_test_knn(X_train, X_test, y_train, y_test):
    param_grid = ParameterGrid({
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    })
    model = KNeighborsClassifier()

    for params in param_grid:
        model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred)

        with mlflow.start_run(run_name=f'KNN_{params}') as run:
            mlflow.log_param("model_type", "KNeighborsClassifier")
            mlflow.log_params(params)
            mlflow.log_metric("f1", score)
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(
                model,
                "knn",
                signature=signature,
                input_example=X_train,
                registered_model_name=model_name
            )
            print(f"Modelo KNN registrado no MLflow! Run ID: {run.info.run_id}")

def train_test_rndf(X_train, X_test, y_train, y_test):
    param_grid = ParameterGrid({
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10]
    })
    model = RandomForestClassifier(random_state=42)

    for params in param_grid:
        model.set_params(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred)

        with mlflow.start_run(run_name=f'RNDF_{params}') as run:
            mlflow.log_param("model_type", "RandomForestClassifier")
            mlflow.log_params(params)
            mlflow.log_metric("f1", score)
            signature = infer_signature(X_train, y_pred)
            mlflow.sklearn.log_model(
                model,
                "rndf",
                signature=signature,
                input_example=X_train,
                registered_model_name=model_name
            )
            print(f"Modelo RNDF registrado no MLflow! Run ID: {run.info.run_id}")

def promote_model(client):
    best_model = None
    best_f1_score = 0

    versions = client.search_model_versions(f"name='{model_name}'")
    for version in versions:
        run_id = version.run_id
        metrics = client.get_run(run_id).data.metrics
        score = metrics.get('f1', 0)

        if score > best_f1_score:
            best_f1_score = score
            best_model = version.version

    if best_model is not None:
        client.transition_model_version_stage(
            name=model_name,
            version=best_model,
            stage="production",
            archive_existing_versions=True
        )
    return best_f1_score, f"{model_name}.{best_model}"

def main():
    df = get_dataset()
    y = df[target_column]
    X = df.drop(columns=[target_column])
    X_train, X_test, y_train, y_test = split_and_clean(X, y)

    client = start_mlflow()
    train_test_knn(X_train, X_test, y_train, y_test)
    train_test_rndf(X_train, X_test, y_train, y_test)

    score, best_model = promote_model(client)
    print(f"Melhor score obtido: {score}")
    return score, best_model

if __name__ == "__main__":
    main()