import os 
import sys
from dataclasses import dataclass
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import logging
import os
from datetime import datetime
import dill
from sklearn.model_selection import GridSearchCV


LOG_FILE = f"{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log"
log_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(log_path, exist_ok = True)

LOG_FILE_PATH = os.path.join(log_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level= logging.INFO,
)

def error_message_detail(error, error_detail:sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error Occurred in python script [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno,str(error)   
    )
    return error_message 

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)  
        self.error_message = error_message_detail(error_message, error_detail = error_detail)

    def __str__(self):
        return self.error_message    

def save_object(file_path, obj):
        try:
            dir_path = os.path.dirname(file_path)

            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, 'wb') as file_obj:
                dill.dump(obj, file_obj)

        except Exception as e:
            raise CustomException(e, sys)       


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Spitting training and test input data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1], 
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                'Random Forest' : RandomForestRegressor(),
                'Decision Tree' : DecisionTreeRegressor(),
                'Gradient Boosting' : GradientBoostingRegressor(),
                'K-Nearest Neighbor' : KNeighborsRegressor(),
                'Linear Regression' : LinearRegression(),
                'CatBoosting' : CatBoostRegressor(),
                'XGB Classifer' : XGBRegressor(),
                'Adaboost' : AdaBoostRegressor()
            }
            params = {
                "Decision Tree":{
                    'criterion':['squared_error', 'freidman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random'],
                    'max_features': ['sqrt', 'log2']
                },

                "Random Forest":{
                    'criterion':['squared_error', 'freidman_mse', 'absolute_error', 'poisson'],
                    'n_estimators': [100, 150, 200, 250, 300, 400, 500],
                    'max_features': ['sqrt', 'log2']
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'absolute_error', 'huber', 'quantile'],
                    'crtiterion':['friedman_mse', 'squared_error'],
                    'n_estimators': [100, 150,  200, 250, 300, 400, 500],
                },
                "Linear Regression":{},
                "K-Nearest Regressor":{
                    'n_neighbour':[5, 6, 7, 8, 9, 10, 12, 13, 14, 15],
                },
                "XGB Regressor":{
                    'learning rate':[0.1, 0.01, 0.05, 0.001],
                    'n_estimators':[8, 10, 12, 13, 14, 15, 18, 20],
                    'iterators':[30, 40, 50, 80, 100]
                },
                "Catboosting Rregressor":{
                    'depth':[6, 8, 10, 12],
                    'n_estimators':[8, 10, 12, 13, 14, 15, 18, 20],
                    'iterators':[30, 40, 50, 80, 100]
                    
                },
                "Adaboost Regressor":{
                    'learning rate':[0.1, 0.01, 0.05, 0.001],
                    'n_estimators':[8, 10, 12, 13, 14, 15, 18, 20],



                }


            }
            def evaluate_models(X_train, y_train, X_test, y_test, models, params, cv = 3, n_jobs = 3, verbose = 3):
                try:
                    report = {}

                    for i in range(len(list(models))):
                        model = list(models.values())[i]
                        para = params[list(models.keys())[i]]

                        gs = GridSearchCV(model, para, cv = cv, n_jobs = n_jobs, verbose =verbose)
                        gs.fit(X_train, y_train)

                        model.set_params(**gs.best_params_)
                        model.fit(X_train, y_train)

                        y_train_pred = model.predict(X_train)
                        y_test_pred = model.predict(X_test)

                        train_model_score = r2_score(y_train, y_train_pred)
                        test_model_score = r2_score(y_test, y_test_pred)

                        report[list(models.keys())[i]] = (test_model_score, gs.best_params_)



                    return report    

                            
                except Exception as e:
                    raise CustomException(e, sys)

            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train, 
                                               X_test = X_test, y_test = y_test, 
                                               models = models, params = params)
            
            # To get the best models score from the above dictionary
            best_model_score = max(sorted(model_report.values()))

            #To get the best model name from the above dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException('No best model found')
            logging.info(f'Best found model on both training and testing dataset{best_model}')

            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
            
