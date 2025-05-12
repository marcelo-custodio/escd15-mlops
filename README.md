# MLOps Pipeline - Churn 

Este repositório contém um pipeline completo de MLOps para prever a evasão de clientes (churn), com versionamento de modelos, monitoramento de drift e reentreinamento automático.

---

## 📊 Dataset

Utilizamos o dataset [Churn for Bank Customers](https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers), disponível no Kaggle. A base contém informações demográficas e de relacionamento com o banco, com o objetivo de prever se um cliente irá sair da instituição.

---

## Pipeline do Projeto

### 1. **Exploração e Pré-processamento**
- Tratamento de valores ausentes
- Normalização e codificação de variáveis
- Split treino/teste
- Scripts: `ml_model/preprocessing.py`

### 2. **Treinamento e Avaliação**
- Modelos utilizados:
  - K-Nearest Neighbors
  - Random Forest
- Métrica: F1 Score
- Rastreado com [MLflow](https://mlflow.org/)
- Script principal: `ml_model/main.py`

### 3. **Versionamento com MLflow**
- Modelos são registrados e versionados automaticamente no MLflow Model Registry.
- O melhor modelo é promovido automaticamente para produção.

### 4. **API com FastAPI**
- Rota `/predict` para realizar predições em tempo real.
- Script da API: `api/main.py`

### 5. **Monitoramento com Evidently AI**
- Verificação de **Data Drift** a cada 1000 requisições.
- Geração de relatório em HTML (`data_drift_report.html`)
- Script: `monitoring/drift_check.py`

### 6. **Re-treinamento Automático**
- Ao detectar drift > 30%, o pipeline executa `main.py` novamente.
- Os novos dados são combinados com o dataset original.

### 7. **Armazenamento de Requisições**
- Requisições são armazenadas em banco SQLite (`requests.db`)
- Script: `monitoring/api_logger.py`

### 8. **Orquestração do MlFlow + Aplicação com Supervisor**
- Retry automático caso MlFlow ou a Aplicação tenham algum problema
- Disponibilização do Backend e interface de ambos em um único container

---

### Instalar dependências

```bash
pip install -r requirements.txt
```

### Execução Local

Estar na pasta do projeto e executar os seguintes comandos no terminal:

- MlFlow: ```mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 --host 0.0.0.0```
- API: ```uvicorn app:app --host 0.0.0.0 --port 8000```

### Execução via Docker

Estar na pasta do projeto e executar os seguintes comandos via terminal:

- Build da Imagem: ```docker build . -t escd15-mlops```
- Execução da aplicação em container: ```docker run -p 5000:5000 -p 8000:8000 escd15-mlops```
