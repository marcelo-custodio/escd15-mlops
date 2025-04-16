from ml_model.preprocessing import get_dataset, split_and_clean
from parameters import target_column
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_test_knn(X_train, X_test, y_train, y_test):
    ...

def train_test_rndf(X_train, X_test, y_train, y_test):
    ...

def main():
    df = get_dataset()
    y = df[target_column]
    X = df.drop(columns=[target_column])
    X_train, X_test, y_train, y_test = split_and_clean(X, y)

    train_test_knn(X_train, X_test, y_train, y_test)
    train_test_rndf(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()