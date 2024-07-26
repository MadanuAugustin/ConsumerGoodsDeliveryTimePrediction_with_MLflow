
import sys
import mysql.connector
import pandas as pd
from src.deliverytime.entity.config_entity import DataIngestionConfig
from src.deliverytime import CustomException, logger




class DataIngestion:
    def __init__(self, config : DataIngestionConfig):
        self.config = config



    # the below method is responsible for fetching the data from the SQL database

    def download_file(self):
        logger.info(f'-------Entered into download_file method------------')
        try:
            data_base_config = {
            'user': 'root',
            'password': '#@augustin#@7',
            'host': 'localhost',
            'database': 'fooddeliverytimeprediction'
            }

            connection = mysql.connector.connect(**data_base_config)

            query = "SELECT * FROM fooddeliverytimeprediction"

            logger.info('---Requesting the SQL database------')

            raw_data = pd.read_sql(query, connection)

            logger.info('------Successfully fetched data from the SQL database----------')

            raw_data.to_csv(self.config.local_data_file, index=False)

            logger.info(f'----------The shape of the Raw data is {raw_data.shape}--------------------')

            connection.close()

        except Exception as e:
            raise CustomException(e, sys)