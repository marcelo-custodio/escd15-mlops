import pandas as pd
from parameters import columns_to_consider, target_column
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler, StandardScaler

def get_dataset():
    # lendo dataset
    df = pd.read_csv('churn.csv')

    # filtrando as colunas a serem utilizadas
    _columns = [column for column in df.columns if column in columns_to_consider.keys()]
    df = df[_columns]

    # tratando as colunas
    for column in df.columns:
        # aplicando schema no csv
        typo = columns_to_consider.get(column, str)
        df[column] = df[column].astype(typo)
        # aplicando limpeza nas colunas do tipo string
        if typo == str:
            df[column] = df[column].str.replace(r'[^a-zA-Z]', '', regex=True).str.lower()
            df[column] = df[column].apply(lambda x: None if str(x) == '' else str(x))
    df.dropna(inplace=True, ignore_index=True)
    return df

def split_and_clean(X, y):
    # categorizando países
    X['Geography'] = LabelEncoder().fit_transform(X['Geography'])
    # categorizando generos
    X['Gender'] = LabelEncoder().fit_transform(X['Gender'])

    # split treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # escalando saldo 0:max >> 0:1
    max_train = X_train['Balance'].max()
    X_train['Balance'], X_test['Balance'] = (
        (X_train['Balance'] / max_train),
        (X_test['Balance'] / max_train)
    )

    # escalando idade
    ss = StandardScaler()
    ss.fit(X_train[['Age']])
    X_train['Age'], X_test['Age'] = (
        ss.transform(X_train[['Age']]),
        ss.transform(X_test[['Age']])
    )

    # escalando anos de lealdade
    ss = StandardScaler()
    ss.fit(X_train[['Tenure']])
    X_train['Tenure'], X_test['Tenure'] = (
        ss.transform(X_train[['Tenure']]),
        ss.transform(X_test[['Tenure']])
    )

    # escalando score min:max >> -10:10
    mms = MinMaxScaler(feature_range=(-10, 10))
    mms.fit(X_train[['CreditScore']])
    X_train['CreditScore'], X_test['CreditScore'] = (
        mms.transform(X_train[['CreditScore']]),
        mms.transform(X_test[['CreditScore']])
    )

    # escalando salario estimado, evitando outliers
    res = RobustScaler()
    res.fit(X_train[['EstimatedSalary']])
    X_train['EstimatedSalary'], X_test['EstimatedSalary'] = (
        res.transform(X_train[['EstimatedSalary']]),
        res.transform(X_test[['EstimatedSalary']])
    )

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df = get_dataset()
    y = df[target_column]
    X = df.drop(columns=[target_column])
    X_train, X_test, y_train, y_test = split_and_clean(X, y)