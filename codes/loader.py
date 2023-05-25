import pandas as pd
import numpy as np
import polars as pl
import logging

from codes import utils, preprocess

class DataLoader():

    def __init__(self, input_path='../input', log_path=None, options={
        'sampling': 5000,
        'split_labels': True,
        'low_mem': False,
        'load_additional': False
    }):

        self.logger = utils.Logger(log_path)
        self.input_path = input_path
        self.options = options
    
    def load(self, ):

        self.logger.info(f'read_csv from {self.input_path}')
        df_test = pd.read_csv(f'{self.input_path}/test.csv')
        df_submission = pd.read_csv(f'{self.input_path}/sample_submission.csv')

        if self.options.get('low_mem'):
            self.logger.info(f'Use low_memory parquet.')
            df_train = pl.read_parquet(f'{self.input_path}/train_low_mem.parquet').drop(["fullscreen", "hq", "music"])    
            df_train = df_train.with_columns(df_train['text'].cast(pl.Utf8))
        else:
            df_train = pl.read_parquet(f'{self.input_path}/train.parquet').drop(["fullscreen", "hq", "music"])
        df_labels = pd.read_parquet(f'{self.input_path}/train_labels.parquet')

        if self.options.get('split_labels'):
            df_labels = preprocess.split_df_labels(df_labels)

        if self.options.get('sampling'):
            n_sample = self.options.get('sampling')
            self.logger.info(f'Sampling data to {n_sample} sessions.')

            sample_session = df_train.get_column('session_id').unique().sample(n_sample).to_list()
            df_train = df_train.filter(pl.col('session_id').is_in(sample_session))
    
            if self.options.get('split_labels'):
                df_labels = df_labels[df_labels['session'].isin(sample_session)]
        
        df_train_additional = None
        df_labels_additional = None
        if self.options.get('load_additional'):
            self.logger.info(f'Load additional data.')
            df_train_additional = pl.read_parquet(f'{self.input_path}/train_additional.parquet')
            df_labels_additional = pd.read_parquet(f'{self.input_path}/train_labels_additional.parquet')

        return df_train, df_test, df_labels, df_submission, df_train_additional, df_labels_additional