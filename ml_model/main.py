from ml_model.preprocessing import get_dataset, split_and_clean
from parameters import target_column
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
import mlflow
import mlflow.sklearn

def start_mlflow():
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    mlflow.set_experiment("customerchurn")
    return

def train_test_knn(X_train, X_test, y_train, y_test):
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "KNeighborsClassifier")

        model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred)
        mlflow.log_metric("f1", score)

        mlflow.sklearn.log_model(model, "knn")
        print(f"Modelo KNN registrado no MLflow! Run ID: {run.info.run_id}")

def train_test_rndf(X_train, X_test, y_train, y_test):
    with mlflow.start_run() as run:
        mlflow.log_param("model_type", "RandomForestClassifier")

        model = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=10, min_samples_split=5)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        score = f1_score(y_test, y_pred)
        mlflow.log_metric("f1", score)

        mlflow.sklearn.log_model(model, "random_forest")
        print(f"Modelo Random Forest registrado no MLflow! Run ID: {run.info.run_id}")

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