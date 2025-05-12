import pandas as pd
import evidently
from evidently import Report
from evidently.presets import DataDriftPreset

# Caminhos dos dados
TRAIN_DATA_PATH = "ml_model/churn.csv"           # Dados de treino originais
PROD_DATA_PATH = "ml_model/production_data.csv"  # Dados recentes simulando produção
REPORT_OUTPUT = "drift_report.html"
DRIFT_THRESHOLD = 0.3
CHECK_INTERVAL = 1000
DB_PATH = "requests.db"

# Carregar os datasets
df_train = pd.read_csv(TRAIN_DATA_PATH)
df_prod = pd.read_csv(PROD_DATA_PATH)

def check_and_retrain():
    conn = sqlite3.connect(DB_PATH)
    df_inputs = pd.read_sql("SELECT * FROM inputs", conn)
    conn.close()

    if len(df_inputs) < CHECK_INTERVAL:
        print(f"Aguardando mais requisições. Atual: {len(df_inputs)}/{CHECK_INTERVAL}")
        return

    df_original = pd.read_csv(TRAIN_DATA_PATH)
    df_inputs = df_inputs[df_original.columns]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=df_original, current_data=df_inputs)
    report.save_html("ml_model/data_drift_report.html")

    result = report.as_dict()
    drifted = [f for f, v in result["metrics"][0]["result"]["features"].items() if v["drift_detected"]]
    drift_ratio = len(drifted) / len(df_original.columns)

    print(f"Data drift detectado em {drift_ratio:.2%} das features.")

    if drift_ratio > DRIFT_THRESHOLD:
        print("Data drift significativo. Re-treinando modelo.")
        combined = pd.concat([df_original, df_inputs], ignore_index=True)
        combined.to_csv(TRAIN_DATA_PATH, index=False)
        df_inputs.to_csv(PROD_DATA_PATH , index=False)
        subprocess.run(["python", "ml_model/main.py"])

        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM inputs")
        conn.commit()
        conn.close()
