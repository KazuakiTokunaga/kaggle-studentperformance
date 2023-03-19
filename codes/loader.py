import pandas as pd
import numpy as np
import polars as pl

from codes import utils

class DataLoader():

    def __init__(self, input_path='../input/', options={
        'sampling': 5000,
        'split_labels': True
    }):

        self.input_path = input_path
        self.options = options
    
    def load(self, ):

        print(f'read_csv from {self.input_path}')
        df_train = pl.read_csv(f'{self.input_path}/train.csv').drop(["fullscreen", "hq", "music"])
        df_test = pl.read_csv(f'{self.input_path}/test.csv')
        df_labels = pl.read_csv(f'{self.input_path}/train_labels.csv')
        df_submission = pd.read_csv(f'{self.input_path}/sample_submission.csv')

        if self.options.get('split_labels'):
            df_labels = preprocess.split_df_labels(df_labels)

        if self.options.get('sampling'):
            n_sample = options.get('sampling')
            sample_session = df.get_column('session_id').unique().sample(n_sample).to_list()
            
            df_train = df.filter(pl.col('session_id').is_in(sample_session))
    
            if self.options.get('split_labels'):
                df_labels = df_labels.filter(pl.col('session').is_in(sample_session))

        return df_train, df_test, df_labels, df_submission