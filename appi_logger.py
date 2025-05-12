import sqlite3
import pandas as pd

def save_to_db(input_data: dict, db_path="requests.db"):
    conn = sqlite3.connect(db_path)
    df = pd.DataFrame([input_data])
    df.to_sql("inputs", conn, if_exists="append", index=False)
    conn.close()
