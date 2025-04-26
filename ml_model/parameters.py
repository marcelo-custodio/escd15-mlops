columns_to_consider = {
    'CreditScore': int,
    'Geography': str,
    'Gender': str,
    'Age': int,
    'Tenure': int,
    'Balance': float,
    'NumOfProducts': int,
    'HasCrCard': int,
    'IsActiveMember': int,
    'EstimatedSalary': float,
    'Exited': int
}

model_name = "churn_classifier"
target_column = 'Exited'
