from src.churn.components import data_tranformation
from src.churn.exception import CustomException
from src.churn.logger import logging
from src.churn.components.data_ingestion import DataIngestionConfig,DataIngestion
import sys
from src.churn.components.data_tranformation import DataTransformation
import pandas as pd
from src.churn.components.model_trainer import ModelTrainer
if __name__=="__main__" :
    try:
        Data_ingestion=DataIngestion()
        train,test,validation=Data_ingestion.initiate_data_ingestion()
        data_tranformation=DataTransformation()
        train_df,test_df,churn_risk_train,churn_risk_test=data_tranformation.initiate_data_transformation(train,validation)
      
        Model_Trainer=ModelTrainer()
        Model_Trainer.initiale_model_trainer(train_df,test_df,churn_risk_train,churn_risk_test)
    except Exception as e:
        raise CustomException(e,sys)
    