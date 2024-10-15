import os
import pickle
import sys

from sklearn.model_selection import GridSearchCV
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
    
#def evaluate_models(x_train,y_train,x_test,y_test,models,param):
   # try:
    #    report={}

     #   for i in range(len(list(models))):
     #        model=list(models.values())[i]
     #        para=param[list(models.keys())[i]]

      #       gs=GridSearchCV(model,param,cv=3)
       #      gs.fit(x_train,y_train)

        #     model.set_params(**gs.best_params_)
         #    model.fit(x_train,y_train)
#
 #            y_train_pred=model.predict(x_train)
  #           y_test_pred=model.predict(x_test)

   #          train_model_score=

    #         report[list(models.keys())[i]]=train_model_score

     #        return report



            



    except Exception as e:
            raise CustomException(e,sys)
     
    
def save_object(file_path,obj):
    try:
            dir_path=os.path.dirname(file_path)
            os.makedirs(dir_path,exist_ok=True)

            with open(file_path,'wb') as file_obj:
                pickle.dump(obj,file_obj)
        

    except Exception as e:
           raise CustomException(e,sys)

