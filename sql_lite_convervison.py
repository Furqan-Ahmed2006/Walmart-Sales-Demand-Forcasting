import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

# MySQL Connection (Source)
mysql_engine = create_engine(f"mysql+mysqlconnector://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}")

# SQL se data read karein
df = pd.read_sql("SELECT * FROM gold_sales_data", con=mysql_engine)

# SQLite Connection (Destination - Yeh file create ho jayegi)
sqlite_conn = sqlite3.connect('walmart_data.db')

# Data ko SQLite mein save karein
df.to_sql('gold_sales_data', sqlite_conn, index=False, if_exists='replace')

sqlite_conn.close()
print("✅ Success! Aapka data 'walmart_data.db' mein transfer ho gaya hai.")