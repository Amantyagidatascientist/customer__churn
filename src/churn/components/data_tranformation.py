import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from scipy.sparse import csr_matrix
from src.churn.utils import save_object
from src.churn.exception import CustomException
from src.churn.logger import logging
import os


@dataclass
class DataTransformationConfig:
    predecessor_obj_file_path1 = os.path.join('artifacts', 'train_data_arr_Content.pkl')
    predecessor_obj_file_path2= os.path.join('artifacts', 'train_target_data_extraction.pkl')
    predecessor_obj_file_path3= os.path.join('artifacts', 'train_data_arr_Collaborative.pkl')



class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            time_data=['joining_date', 'last_visit_time']
            numaric_columns=['customer_id','age','days_since_last_login','avg_time_spent',
                             'avg_transaction_value','points_in_wallet']
            catigorcal_columns=['gender','region_category','membership_category','joined_through_referral'
                                ,'preferred_offer_types','medium_of_operation','internet_option',
                                'used_special_discount','offer_application_preference','past_complaint','complaint_status','feedback']
            drop_columns=['Name','security_no','referral_id']
            
        except CustomException as e:
            logging.error("Custom Exception occurred", exc_info=True)
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self,train,test):
        try:
            logging.info("Read the train data and test data files")
            train = pd.read_csv(train)
            test = pd.read_csv(test)
             
        except CustomException as e:
            logging.error("Custom Exception occurred", exc_info=True)
            raise CustomException(e, sys)

