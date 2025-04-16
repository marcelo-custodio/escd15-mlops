from ml_model.preprocessing import get_dataset, split_and_clean
from parameters import target_column

def main():
    df = get_dataset()
    y = df[target_column]
    X = df.drop(columns=[target_column])
    X_train, X_test, y_train, y_test = split_and_clean(X, y)

if __name__ == "__main__":
    main()