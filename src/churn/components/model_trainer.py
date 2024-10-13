import os
import sys
from dataclasses import dataclass

from catboost import CatBoostClassifier
from numpy import square
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.churn.exception import CustomException
from src.churn.logger import logging
from src.churn.utils import read_sql_data,evaluate_models

@dataclass

class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def initiale_model_trainer(self,train,test,target,target_test):
        try:
            models={
                "Random Forest" : RandomForestClassifier(),
                "Decision Tree" : DecisionTreeClassifier(),
                "Gradient Boosting" : GradientBoostingClassifier(),
                "XG Boosting" : XGBClassifier(),
                "catBoosting" : CatBoostClassifier(),
                "AdaBoosting" : AdaBoostClassifier(),

            }
            params={
                "Decision Tree": {
                    "critersion":["squared_error","friedman_mse","absolute_error","poisson"],
                    "splitter":["best","random"],
                    "max_features":['sqrt','log2']
                },
                "Random Forest" :{
                    "critersion":["squared_error","friedman_mse","absolute_error","poisson"],
                    "n_estimators":[8,16,32,64,128,256],
                    "max_features":['sqrt','log2']

                },
                "Gradient Boosting":{
                    "loss":["squared_error","huber","absolute_error","quantile"],
                    "learning_rate":[0.1,0.01,0.05,.001],
                    "subsample": [0.6,0.7,0.75,0.8,0.85,0.9],
                    "criterion":["squared_error","friedman_mse"],
                    "max_features":['auto','sqrt','log2'],
                    "n_estimators":[8,16,32,64,128,256]
                },
                
                "XGBRegressor":{
                    "learning_rate":[0.1,0.01,0.05,0.001],
                    "n_estimators":[8,16,32,64,128,256],

                },
                "CatBoosting":{
                    "depth":[6,8,10],
                    "learing_rate":[0.01,0.05,0.001],
                    "iterations":[30,50,100]
                },
                "AdaBoost":{
                    "Learning_rate":[0.1,0.01,0.5,0.001],
                    "loss":['linear',"square","exponential"],
                    "n_estimators":[8,16,32,64,128,256]
                }


                }
            
            model_report:dict=evaluate_models(train,test,target,target_test,models,params)



            


        except Exception as e:
            raise CustomException(e,sys)
     