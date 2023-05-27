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
        'load_additional': False,
        'additional_grp2_3': False,
        'exclude_low_index': True
    }):

        self.logger = utils.Logger(log_path)
        self.input_path = input_path
        self.options = options
    
    def load(self, ):

        self.logger.info(f'read_csv from {self.input_path}')
        df_test = pd.read_csv(f'{self.input_path}/test.csv')
        df_submission = pd.read_csv(f'{self.input_path}/sample_submission.csv')

        if self.options.get('load_additional'):
            if self.options.get('additional_grp2_3'):
                self.logger.info('Use train data with additional grp2-3.')
                df_train = pl.read_parquet(f'{self.input_path}/train_additional_grp2_3.parquet').drop(["fullscreen", "hq", "music"])    
                df_labels = pd.read_parquet(f'{self.input_path}/train_labels_additional_grp2_3.parquet')
            else:
                self.logger.info('Use train data with additional all grps.')
                df_train = pl.read_parquet(f'{self.input_path}/train_additional.parquet').drop(["fullscreen", "hq", "music"])    
                df_labels = pd.read_parquet(f'{self.input_path}/train_labels_additional.parquet')

            df_train = df_train.filter(~pl.col('session_id').is_in([
                22080308213749430, 
                22080309230203028, # grp0-4にログがないが正解ラベルがあるユーザー
                22080409272258496 # grp0-4にログがあるがgrp5-12のログがない
            ])) 

            if self.options.get('exclude_low_index'):
                self.logger.info('Exclude session_id with little idx in grp0-4.')
                df_train = df_train.filter(~pl.col('session_id').is_in([
                    19100214524945590, # grp0-4のidxが少なすぎるユーザー
                    19100320360093440,
                    19100612343439052,
                    19100618271938656,
                    19110018402009590,
                    19110108401614536,
                    19110210025294616,
                    19110212483080596,
                    19110213384936910,
                    19110215033958504,
                    19110215480952410,
                    19110216043329196,
                    19110309164363510,
                    19110309345174636,
                    19110412301616656,
                    19110414404458970,
                    19110416231634000,
                    19110417425108044,
                    19110510032096820,
                    19110510153950050,
                    19110512114758492,
                    19110612431902212,
                    19110617311137200,
                    19110617575884412,
                    19110618472816120,
                    19110620094779560,
                    20000009553968380,
                    20000108271346350,
                    20000110233102610,
                    20000111591701276,
                    20000113443865924,
                    20000116465787984,
                    20000119552035170,
                    20000209283797016,
                    20000209294035700,
                    20000210455852770,
                    20000212430437836,
                    20000213003172732,
                    20000215420532484,
                    20000216234809676,
                    20000216303627120,
                    20000218261154504,
                    20000307364487652,
                    20000310184288120,
                    20000315302498376,
                    20000317520478710,
                    20000319110479440,
                    20000410310717724,
                    20000411183586570,
                    20000413000865356,
                    20000414315230140,
                    20000417035372524,
                    20000420481726936,
                    20000507392867696,
                    20000508350346600,
                    20000510451933204,
                    20000511534011904,
                    20000512574237684,
                    20000516354763496,
                    20000516432436480,
                    20000610030074630,
                    20010010591575224,
                    20010014300257212,
                    20010117304102336,
                    20010117540662604,
                    20010208111864744,
                    20010210294483344,
                    20010214032335910,
                    20010214272061830,
                    20010219232632280,
                    20010309262398316,
                    20010309580637056,
                    20010310421154824,
                    20010310455043956,
                    20010311023888064,
                    20010312070645904,
                    20010313375456150,
                    20010315265542164,
                    20010315401679250,
                    20010318302581690,
                    20010408380084740,
                    20010408440504188,
                    20010409291392000,
                    20010410354911132,
                    20010411431458016,
                    20010414274095830,
                    20010415201684004,
                    20010507564251532,
                    20010511192982860,
                    20010511334749560,
                    20010621275608164,
                    22020313441822130,
                    22020315014597268,
                    22020414542749764,
                    22020415021534064,
                    22020418590200540,
                    22020512013349384,
                    22020512230669090,
                    22020512480608868,
                    22020521473162720,
                    22030019535890064,
                    22030212301868160,
                    22030308463003490,
                    22030310195385196,
                    22030313160694228,
                    22030408520209750,
                    22030410250195188,
                    22030413002248670,
                    22030413003331430,
                    22030413012291690,
                    22030413260708372,
                    22030507365071148,
                    22030508453791484,
                    22030509414581196,
                    22030513400779844,
                    22030514385129376,
                    22030613011583980,
                    22040213484134620,
                    22040213545272610,
                    22040214421623936,
                    22040218123906584,
                    22040308005979988,
                    22040313294300140,
                    22040321063335300,
                    22040415553210292,
                    22040509343758956,
                    22040510395257076,
                    22050414565022704,
                    22060419431331420,
                    22060614475661172,
                    22080118465325244,
                    22080214034707856,
                    22080314363696676,
                    22080412362591136,
                    22080512144441190,
                    22090108323257140,
                    22090111233286748,
                    22090112073077550,
                    22090112313380760,
                    22090208312750650,
                    22090214294749396,
                    22090315001028790,
                    22090320161270460,
                    22090408470758836,
                    22090409571790092,
                    22090412044913828,
                    22090510544696396,
                    22090514231496576,
                    22100010504415190,
                    22100214125501220,
                    22100312093729860,
                    22100312205668130,
                    22100313040242244,
                    22100315262004190,
                    22100610464941670
                ]))
        
        elif self.options.get('low_mem'):
            self.logger.info(f'Use low_memory parquet.')
            df_train = pl.read_parquet(f'{self.input_path}/train_low_mem.parquet').drop(["fullscreen", "hq", "music"])    
            df_train = df_train.with_columns(df_train['text'].cast(pl.Utf8))
            df_labels = pd.read_parquet(f'{self.input_path}/train_labels.parquet')

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


        return df_train, df_test, df_labels, df_submission