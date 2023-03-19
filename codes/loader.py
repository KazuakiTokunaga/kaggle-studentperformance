import pandas as pd
import numpy as np

class DataLoader():

    def __init__(self, input_path='../input/', dataset_path='', options={'process_labels': True}):
        
        self.input_path = input_path
        self.df_train = pd.read_csv(f'{self.input_path}/train.csv')
        self.df_test = pd.read_csv(f'{self.input_path}/test.csv')
        self.df_labels = pd.read_csv(f'{self.input_path}/train_labels.csv')
        self.df_submission = pd.read_csv(f'{self.input_path}/sample_submission.csv')

        if options.get('process_labels'):
            self.df_labels['session'] = self.df_labels.session_id.apply(lambda x: int(x.split('_')[0]) )
            self.df_labels['q'] = self.df_labels.session_id.apply(lambda x: int(x.split('_')[-1][1:]) )



