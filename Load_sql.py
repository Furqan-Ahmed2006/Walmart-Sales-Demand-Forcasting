import os
import logging
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
df=pd.read_csv("Demand_Forcasting/master_cleaned_data.csv")
logging.basicConfig(
    filename='pipeline.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
def load_data_to_mysql(df):
    try:
        load_dotenv()
        user = os.getenv('DB_USER')
        password = os.getenv('DB_PASSWORD')
        host = os.getenv('DB_HOST')
        db_name = os.getenv('DB_NAME')
        connection_url = f'mysql+mysqlconnector://{user}:{password}@{host}/{db_name}'
        engine = create_engine(connection_url)
        logging.info("Attempting to connect to MySQL...")
        print("Connecting to MySQL...")
        df.to_sql('gold_sales_data', con=engine, if_exists='replace', index=False, chunksize=10000)
        logging.info(f"Success: {len(df)} rows loaded into 'gold_sales_data' table.")
        print("Data Loaded To Database Successfully")
    except Exception as e:
        logging.error(f"Failed to load data: {str(e)}")
        print(f"Error: {e}. Check pipeline.log for full details.")
load_data_to_mysql(df)