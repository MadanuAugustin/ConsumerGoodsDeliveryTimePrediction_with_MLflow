




import sys
import joblib
import tensorflow
import numpy as np
import pandas as pd
from pathlib import Path
from src.deliverytime import logger, CustomException



class PredictionPipeline:
    def __init__(self):
        # self.model = joblib.load(Path('model//model.keras'))
        self.model = tensorflow.keras.models.load_model("model//model.keras")
        self.preprocessorObj = joblib.load(Path('model//preprocessor_obj.joblib'))


    # the below method takes the data from the user to predict

    def predictDatapoint(self, data):
        
        try:

            data_df = data.rename(columns = {0 : 'delivery_person_age', 1 : 'delivery_person_ratings', 2 : 'Distance'})
            
            print(data_df)

            transformed_numeric_cols = self.preprocessorObj.transform(data_df)

            logger.info(f'---------Below is the transformed user input----------------')

            print(transformed_numeric_cols)

            prediction = self.model.predict(transformed_numeric_cols)

            logger.info(f'-----------Below output is predicted by the model---------------')

            print(prediction)

            return prediction
        
        
        except Exception as e:
            raise CustomException(e, sys)