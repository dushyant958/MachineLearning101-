import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import sys
import logging
import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging
import os
from datetime import datetime
import sys
import logging
import dill

import logging
import os
from datetime import datetime
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
            def evaluate_models(X_train, y_train, X_test, y_test, models):
                try:
                    report = {}

                    for i in range(len(list(models))):
                        model = list(models.values())[i]
                        model.fit(X_train, y_train)

                        y_train_pred = model.predict(X_train)

                        y_test_pred = model.predict(X_test)

                        train_model_score = r2_score(y_train, y_train_pred)

                        test_model_score = r2_score(y_test, y_test_pred)

                        report[list(models.keys())[i]] = test_model_score

                    return report    
                except Exception as e:
                    raise CustomException(e, sys)

            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train, 
                                               X_test = X_test, y_test = y_test, 
                                               models = models)
            
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
        

@dataclass
class DataTransformationConfig():
    preprocessor_ob_file_path = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns = ['writing score', 'reading score']
            categorical_columns = [
                'gender',
                'race/ethnicity',
                'parental level of education',
                'lunch',
                'test preparation course'
            ]
            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('OneHotEncoder', OneHotEncoder(sparse_output=True)),
                    ('scaler', StandardScaler(with_mean=False))

                ]
            )
            logging.info('Numerical columns standard scaling completed')

            logging.info('Categorical columns encoding completed')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try :
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading train and test data completed')

            logging.info('Obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math score'
            numerical_columns = ['writing score', 'reading score']

           
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f'Applying preprocessing object on training dataframe and testing dataframe'

            )
            input_featue_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_featue_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_featue_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_featue_test_arr, np.array(target_feature_test_df)
            ]

            logging.info('Saved preprocssing object')

            save_object (
                file_path = self.data_transformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,

            )

        except Exception as e:
            raise CustomException(e, sys)
        
            
    
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method or component')
        try:
            df = pd.read_csv('D:\\ml_project\\notebook\\data\\studs.csv.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Train Test Split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Ingestion of the data is completed')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)
            
if __name__ == "__main__":  # Fixed this line
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()  

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))

