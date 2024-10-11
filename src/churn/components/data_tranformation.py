import sys
from dataclasses import dataclass
from xml.etree.ElementInclude import include
import pandas as pd
import numpy as np
from pygit2 import Passthrough
from sklearn import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from scipy.sparse import csr_matrix
from src.churn.utils import save_object
from src.churn.exception import CustomException
from src.churn.logger import logging
from sklearn.preprocessing import OneHotEncoder
import os


@dataclass
class DataTransformationConfig:
    predecessor_obj_file_path1 = os.path.join('artifacts', 'train_df.pkl')
    predecessor_obj_file_path2= os.path.join('artifacts', 'test_df.pkl')
    predecessor_obj_file_path3= os.path.join('artifacts', 'valdtion.pkl')
    predecessor_obj_file_path4= os.path.join('artifacts', 'target.pkl')
    predecessor_obj_file_path5= os.path.join('artifacts', 'target_test.pkl')


class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        return X.drop(columns=self.columns,axis=1)
    
class dropErrorAndConvertDate(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass

    def fit(self,x,y=None):
        return self
    
    def transform(self,x,y=None):
        
        x['avg_frequency_login_days']=x['avg_frequency_login_days'].apply(lambda n:0 if n=='Error' else n)


        x['joining_date']=pd.to_datetime(x['joining_date'])
        x['joining_day']=x['joining_date'].dt.day
        x['joining_month'] = x['joining_date'].dt.month
        x['joining_year'] = x['joining_date'].dt.year


        x['last_visit_time']=pd.to_datetime(x['last_visit_time'])
        x['last_visit_time_hour']=x['last_visit_time'].dt.hour
        x['last_visit_time_minutes']=x['last_visit_time'].dt.minute

        return x



class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            
            numaric_columns=['age','days_since_last_login','avg_time_spent',
                             'avg_transaction_value','points_in_wallet','avg_frequency_login_days']
            categorical_columns=['gender','region_category','membership_category','joined_through_referral'
                                ,'preferred_offer_types','medium_of_operation','internet_option',
                                'used_special_discount','offer_application_preference','past_complaint','complaint_status','feedback']
            columns_drop=['customer_id','Name','security_no','referral_id','joining_date','last_visit_time']
            
            logging.info(f"categorical_columns: {categorical_columns}")
            logging.info(f"Numeric columns: {numaric_columns}")
            logging.info(f"Columns to drop: {columns_drop}")
            numeric_pipeline = Pipeline(steps=[
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', StandardScaler())
            ])

            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown',add_indicator=True)),
                ('OneHotEncoder',OneHotEncoder(drop='if_binary'))
                ])

            columns_transformer=ColumnTransformer(transformers=[('numeric_pipeline',numeric_pipeline,numaric_columns),
                                                                ('categorical_pipeline',categorical_pipeline,categorical_columns)],
                                                                remainder='passthrough')
            
            sub_pipeline=Pipeline(steps=[
                                         ('dropErrorAndConvertDate',dropErrorAndConvertDate()),
                                         ('DropColumns',DropColumns(columns=columns_drop)),
                                         ('columns_transformer',columns_transformer)
                                         ])
            


            return sub_pipeline
   
        except CustomException as e:
            logging.error("Custom Exception occurred", exc_info=True)
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self,train,test):
        try:
            logging.info("Read the train data and test data files")
            train = pd.read_csv(train)
            test = pd.read_csv(test)
            logging.info(f"train shape: {train.shape}")
            logging.info(f"test  shape: {test.shape}")
            churn_risk_test=test['churn_risk_score']
            test=test.drop(columns=['churn_risk_score'],axis=1)


            churn_risk_train=train['churn_risk_score']
            train=train.drop(columns=['churn_risk_score'],axis=1)

            transform_fun=self.get_data_transformer_object()
            train_df=transform_fun.fit_transform(train)
            test_df=transform_fun.transform(test)
            train_df=pd.DataFrame(train_df)
            test_df=pd.DataFrame(test_df)
            logging.info(f"train_df: {train_df.shape}")
            logging.info(f"test_df: {test_df.shape}")


            save_object(file_path=self.data_tranformation_config.predecessor_obj_file_path1,
                        obj=train_df)
            save_object(file_path=self.data_tranformation_config.predecessor_obj_file_path2,
                        obj=test_df)           
            save_object(file_path=self.data_tranformation_config.predecessor_obj_file_path4,
                        obj=churn_risk_train)
            save_object(file_path=self.data_tranformation_config.predecessor_obj_file_path5,
                        obj=churn_risk_test)

            return (train_df,
                    test_df,
                    churn_risk_train,
                    churn_risk_test)
        

             
        except CustomException as e:
            logging.error("Custom Exception occurred", exc_info=True)
            raise CustomException(e, sys)

