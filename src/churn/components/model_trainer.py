import os
import sys
from dataclasses import dataclass
import numpy as np

from catboost import CatBoostClassifier
from numpy import square
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier,RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV


from src.churn.exception import CustomException
from src.churn.logger import logging
from src.churn.utils import save_object

@dataclass

class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    
    def evaluate_models(self,train,test,target,target_test,models,param):
        report={}

        

        random_search = RandomizedSearchCV(
                estimator=models,
                param_distributions=param,
                n_iter=50, 
                scoring='accuracy',  
                cv=5,  
                verbose=2,  
                random_state=42,  
                n_jobs=-1  
)
        random_search.fit(train,target)

        cv_results = random_search.cv_results_['mean_test_score']
        valid_scores = [score for score in cv_results if not np.isnan(score)]

        best_score_idx = np.argmax(valid_scores)
        best_params = random_search.cv_results_['params'][best_score_idx]
        best_score = valid_scores[best_score_idx]

        models.set_params(**best_params)
        models.fit(train, target)

        y_train_pred=models.predict(train)
        y_test_pred=models.predict(test)

        test_model_score=accuracy_score(y_test_pred,target_test)
        train_model_score=accuracy_score(y_train_pred,target)

        report["Gradient_Boosting_Classifier"] = test_model_score

        return report, best_params, best_score
    
    def initiale_model_trainer(self,train,test,target,target_test):
        try:
            models = GradientBoostingClassifier()

            params = {

                   
                    'loss': ['log_loss'],  
                    'learning_rate': [0.01, 0.05, 0.1], 
                    'n_estimators': [100, 200, 500, 1000], 
                    'max_depth': [3, 5, 7, 9],  
                    'min_samples_split': [2, 5, 10],  
                    'min_samples_leaf': [1, 2, 5], 
                    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],  
                    'max_features': ['auto', 'sqrt', 'log2', None]  
                }
                   
                   
                
            
            model_report, best_params, best_model_score=self.evaluate_models(train,test,target,target_test,models,params)

            

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info("Best model is found")

            logging.info(f"Best model is found with accuracy: {best_model_score}")

            predicted = models.predict(test)
            confusionmatrix = confusion_matrix(target_test, predicted)


            save_object(file_path=self.model_trainer_config.trained_model_file_path,
                        obj=models)
            
            pridicted=models.predict(test)

            confusionmatrix=confusion_matrix(target_test,pridicted)

            return models,confusionmatrix
          
        except Exception as e:
            raise CustomException(e,sys)
     