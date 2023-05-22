import pandas as pd
import numpy as np
import polars as pl
import logging

from codes import utils, preprocess

logger = utils.Logger("/kaggle/test20230522")

class DataLoader():

    def __init__(self, input_path='../input', options={
        'sampling': 5000,
        'split_labels': True,
        'parquet': True
    }):

        self.input_path = input_path
        self.options = options
    
    def load(self, ):

        logger.info(f'read_csv from {self.input_path}')
        df_test = pd.read_csv(f'{self.input_path}/test.csv')
        df_submission = pd.read_csv(f'{self.input_path}/sample_submission.csv')

        if self.options.get('parquet'):
            df_train = pl.read_parquet(f'{self.input_path}/train.parquet').drop(["fullscreen", "hq", "music"])
            df_labels = pd.read_parquet(f'{self.input_path}/train_labels.parquet')
        else:
            df_train = pl.read_csv(f'{self.input_path}/train.csv').drop(["fullscreen", "hq", "music"])
            df_labels = pd.read_csv(f'{self.input_path}/train_labels.csv')

        if self.options.get('split_labels'):
            df_labels = preprocess.split_df_labels(df_labels)

        if self.options.get('sampling'):
            n_sample = self.options.get('sampling')
            logger.info(f'Sampling data to {n_sample} sessions.')

            sample_session = df_train.get_column('session_id').unique().sample(n_sample).to_list()
            df_train = df_train.filter(pl.col('session_id').is_in(sample_session))
    
            if self.options.get('split_labels'):
                df_labels = df_labels[df_labels['session'].isin(sample_session)]

        return df_train, df_test, df_labels, df_submission