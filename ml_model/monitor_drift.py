import pandas as pd
import evidently
from evidently import Report
from evidently.presets import DataDriftPreset

# Caminhos dos dados
TRAIN_DATA_PATH = "ml_model/churn.csv"           # Dados de treino originais
PROD_DATA_PATH = "ml_model/production_data.csv"  # Dados recentes simulando produção
REPORT_OUTPUT = "drift_report.html"

# Carregar os datasets
df_train = pd.read_csv(TRAIN_DATA_PATH)
df_prod = pd.read_csv(PROD_DATA_PATH)

# Alinhar colunas se necessário (garante que estejam na mesma ordem)
df_prod = df_prod[df_train.columns]

# Separar features e target
TARGET_COLUMN = "Exited"
ref_data = df_train.copy()
prod_data = df_prod.copy()

# Gerar relatório com Evidently
report = Report(metrics=[
    DataDriftPreset()
])

report.run(reference_data=ref_data, current_data=prod_data)

# Salvar relatório
report.save_html(REPORT_OUTPUT)
print(f"Relatório de monitoramento salvo em: {REPORT_OUTPUT}")
