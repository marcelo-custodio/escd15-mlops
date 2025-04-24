from ml_model.preprocessing import get_dataset, split_and_clean
from parameters import target_column
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

def start_mlflow():
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    mlflow.set_experiment("customerchurn")
    return

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
                registered_model_name="KNeighborsClassifier"
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
                registered_model_name="RandomForestClassifier"
            )
            print(f"Modelo RNDF registrado no MLflow! Run ID: {run.info.run_id}")

def main():
    df = get_dataset()
    y = df[target_column]
    X = df.drop(columns=[target_column])
    X_train, X_test, y_train, y_test = split_and_clean(X, y)

    start_mlflow()
    train_test_knn(X_train, X_test, y_train, y_test)
    train_test_rndf(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()