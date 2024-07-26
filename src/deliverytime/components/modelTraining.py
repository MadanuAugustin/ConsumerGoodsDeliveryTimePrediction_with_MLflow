

import sys
import os
import joblib
import pandas as pd
import tensorflow
from src.deliverytime.entity.config_entity import ModelTrainerConfig
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from src.deliverytime import logger, CustomException
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



class ModelTrainer:
    def __init__(self, config : ModelTrainerConfig):
        self.config = config


    def initiate_model_training(self):

        try:
            
            train_data = pd.read_csv(self.config.train_data_path)
            test_data = pd.read_csv(self.config.test_data_path)

            train_x = train_data.drop([self.config.target_column], axis = 1)
            test_x = test_data.drop([self.config.target_column], axis = 1)
            

            train_y = train_data[[self.config.target_column]]
            test_y = test_data[[self.config.target_column]]

            model = Sequential()
            model.add(LSTM(self.config.lstm_layer_1, return_sequences = True, input_shape = (train_x.shape[1], 1)))
            model.add(LSTM(self.config.lstm_layer_2, return_sequences = False))
            model.add(Dense(self.config.Dense_layer_1))
            model.add(Dense(self.config.Denser_layer_2))
            model.summary()

            model.compile(optimizer = self.config.optimizer, loss = self.config.loss)

            logger.info(f'------------------Model Training started----------------')

            checkpoint_callback = ModelCheckpoint(
                filepath= self.config.model_name,      
                monitor='loss',           
                save_best_only=True,           
                verbose=1                       
            )

            early_stopping_callback = EarlyStopping(
                monitor='loss',             
                patience=3,                    
                verbose=1                       
            )

            history = model.fit(train_x, train_y,  batch_size = self.config.batch_size, epochs = self.config.Epochs, callbacks = [checkpoint_callback, early_stopping_callback])

            logger.info(f'------------------Model Training completed----------------')

            # joblib.dump(model, os.path.join(self.config.root_dir, self.config.model_name))

            # model.save(self.config.model_name)

            logger.info(f'-------------------Model saved as a joblib file----------------')


        except Exception as e:
            raise CustomException(e, sys)