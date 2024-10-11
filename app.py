from src.churn.components import data_tranformation
from src.churn.exception import CustomException
from src.churn.logger import logging
from src.churn.components.data_ingestion import DataIngestionConfig,DataIngestion
import sys
from src.churn.components.data_tranformation import DataTransformation
import pandas as pd
if __name__=="__main__" :
    try:
        Data_Ingestion=DataIngestion()
        train,test,validation=Data_Ingestion.initiate_data_ingestion()
        data_tranformation=DataTransformation()
        r,_,_,_=data_tranformation.initiate_data_transformation(train,validation)
        
        
        print(r.info())
    except Exception as e:
        raise CustomException(e,sys)
    