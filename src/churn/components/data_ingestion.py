import os
import random
import sys
from src.churn.logger import logging
import pandas as pd
from src.churn.exception import CustomException
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.churn.utils import  read_sql_data


class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    validation_data_path: str = os.path.join('artifacts', 'validation.csv')

class DataIngestion:
        def __init__(self):
              self.ingestion_config = DataIngestionConfig()

        
        def initiate_data_ingestion(self):
            try:
                  logging.info("read the completed database")
                  train,test=read_sql_data()
                  os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

                  test.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

                  train,validation=train_test_split(train,test_size=0.20,random_state=42)

                  train.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
                  test.to_csv(self.ingestion_config.validation_data_path,index=False,header=True)

                  logging.info("data_ingestion is completed")

                  return (
                        self.ingestion_config.test_data_path,
                        self.ingestion_config.train_data_path,
                        self.ingestion_config.validation_data_path

                  )

                
            except Exception as e:
                      raise CustomException(e,sys)
              
        
         


