


import pandas as pd
import joblib
import mlflow
import os
import mlflow.sklearn
from pathlib import Path
import tensorflow
from src.deliverytime.entity.config_entity import ModelEvaluationConfig
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.deliverytime.utils.common import save_json
from src.deliverytime import logger, CustomException


class ModelEvaluation:
    def __init__(self, config : ModelEvaluationConfig):
        self.config = config


    
    def eval_metrics(self, actual, pred):
        r2score = r2_score(actual, pred)
        meanabsoluteerror = mean_absolute_error(actual, pred)
        meansquarederror = mean_squared_error(actual, pred)
        return r2score, meanabsoluteerror, meansquarederror
    


    def log_into_mlflow(self):

        logger.info(f'-----------Entered log_into_mlflow function----------------')

        test_data = pd.read_csv(self.config.test_data_path)

        model = tensorflow.keras.models.load_model(self.config.model_path)

        logger.info(f'-----------successfully loaded model joblib--------------------------')

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        os.environ["MLFLOW_TRACKING_URI"]='https://dagshub.com/augustin7766/ConsumerGoodsDeliveryTimePrediction_with_MLflow.mlflow'
        os.environ["MLFLOW_TRACKING_USERNAME"]="augustin7766"
        os.environ["MLFLOW_TRACKING_PASSWORD"]="8a01ee4bec043666cf3ced22edc7d308526b4b42"

        mlflow.set_experiment('first_exp_01')

        with mlflow.start_run():

            logger.info(f'------------------mlflow function started--------------------------------')

            predicted = model.predict(test_x)

            (r2score, meanabsoluteerror, meansquarederror) = self.eval_metrics(test_y, predicted)

            scores = {'r2score' : r2score, 'meanabsoluteerror' : meanabsoluteerror, 'meansquarederror' : meansquarederror}

            save_json(path = Path(self.config.metric_file_name), data = scores)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric('r2score', r2score)
            mlflow.log_metric('meanabsoluteerror', meanabsoluteerror)
            mlflow.log_metric('meansquarederror', meansquarederror)

            mlflow.sklearn.log_model(model, 'model', registered_model_name = 'LSTM')

            logger.info(f'------------------------mlflow function completed-----------------------')
