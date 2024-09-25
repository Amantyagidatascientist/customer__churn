from src.churn.exception import CustomException
from src.churn.logger import logging
from src.churn.components.data_ingestion import DataIngestionConfig,DataIngestion
import sys
if __name__=="__main__" :
    try:
        Data_Ingestion=DataIngestion()
        Data_Ingestion.initiate_data_ingestion()
    except Exception as e:
        raise CustomException(e,sys)
    