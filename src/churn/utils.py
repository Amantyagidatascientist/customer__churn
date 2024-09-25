import os
import sys
from src.churn.logger import logging
import pandas as pd
from src.churn.exception import CustomException
from dataclasses import dataclass
from dotenv import load_dotenv
from sqlalchemy import create_engine


load_dotenv()

host=os.getenv('host')
user=os.getenv('user')
password=os.getenv('password')
db=os.getenv('database')
port = '3306'


def read_sql_data():
    logging.info("reading sql database started")
    try:
        
        engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}:{port}/{db}")
        query1 = "SELECT * FROM train"
        query2 = "SELECT * FROM test"

        train = pd.read_sql(query1, engine)
        test = pd.read_sql(query2, engine)
        
        logging.info("reading is  sql database end")

        print(test.head())



        return train , test

    except Exception as e:
        raise CustomException(e,sys)

