import numpy as np
import pandas as pd
import polars as pl
import gc
import datetime

from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from codes import utils, loader, preprocess

logger = utils.Logger()


class Runner():

    def __init__(self, run_fold_name = 'run',
        input_path='/kaggle/input/student-performance-my', 
        load_options={
            'sampling': 5000,
            'split_labels': True,
            'parquet': True
        },
        validation_options={
            'n_fold': 5
        }):

        self.run_fold_name = run_fold_name
        self.input_path = input_path
        self.load_options = load_options
        
        self.validation_options = validation_options
        self.n_fold = validation_options.get('n_fold')


    def load_dataset(self, ):
        dataloader = loader.DataLoader(input_path=self.input_path, options=self.load_options)
        self.df_train, self.df_test, self.df_labels, self.df_submission = dataloader.load()


    def delete_df_train(self, ):
        logger.info('Delete df_train and run a full collection.')

        del self.df_train
        gc.collect()
    

    def engineer_features(self, ):
        logger.info('Start engineer features.')

        # レコード単位の処理
        columns = [

            pl.col("page").cast(pl.Float32),
            (
                (pl.col("elapsed_time") - pl.col("elapsed_time").shift(1)) 
                .fill_null(0)
                .clip(0, 1e9)
                .over(["session_id", "level_group"])
                .alias("elapsed_time_diff")
            ),
            (
                (pl.col("screen_coor_x") - pl.col("screen_coor_x").shift(1)) 
                .abs()
                .over(["session_id", "level_group"])
                .alias("location_x_diff") 
            ),
            (
                (pl.col("screen_coor_y") - pl.col("screen_coor_y").shift(1)) 
                .abs()
                .over(["session_id", "level_group"])
                .alias("location_y_diff") 
            ),
            pl.col("fqid").fill_null("fqid_None"),
            pl.col("text_fqid").fill_null("text_fqid_None")
        ]

        self.df_train = self.df_train.with_columns(columns)

        # グループごとに分割
        df1 = self.df_train.filter(pl.col("level_group")=='0-4')
        df2 = self.df_train.filter(pl.col("level_group")=='5-12')
        df3 = self.df_train.filter(pl.col("level_group")=='13-22')

        # sessionごとにまとめる
        self.df1 = preprocess.feature_engineer_pl(df1, grp='0-4', use_extra=True, feature_suffix='')
        logger.info(f'df1 done: {self.df1.shape}')
        self.df2 = preprocess.feature_engineer_pl(df2, grp='5-12', use_extra=True, feature_suffix='')
        logger.info(f'df2 done: {self.df2.shape}')
        self.df3 = preprocess.feature_engineer_pl(df3, grp='13-22', use_extra=True, feature_suffix='')
        logger.info(f'df3 done: {self.df3.shape}')
    

    def run_validation(self, ):

        if type(self.df1) == pl.DataFrame:
            logger.info('Convert polars df to pandas df.')

            self.df1 = utils.pl_to_pd(self.df1)
            self.df2 = utils.pl_to_pd(self.df2)
            self.df3 = utils.pl_to_pd(self.df3)
        
        self.df1 = self.df1.fillna(-1)
        self.df2 = self.df2.fillna(-1)
        self.df3 = self.df3.fillna(-1)

        self.ALL_USERS = self.df1.index.unique()
        user_cnt = len(self.ALL_USERS)
        logger.info(f'We will train with {user_cnt} users info')

        gkf = GroupKFold(n_splits=self.n_fold)
        self.oof = pd.DataFrame(data=np.zeros((len(self.ALL_USERS),18)), index=self.ALL_USERS)
        models = {}

        logger.info(f'Start validation with {self.n_fold} folds.')
        for i, (train_index, test_index) in enumerate(gkf.split(X=self.df1, groups=self.df1.index)):

            logger.info(f'Fold {i}')
            # ITERATE THRU QUESTIONS 1 THRU 18
            for t in range(1,19):
                
                # USE THIS TRAIN DATA WITH THESE QUESTIONS
                if t<=3: 
                    grp = '0-4'
                    df = self.df1
                elif t<=13: 
                    grp = '5-12'
                    df = self.df2
                elif t<=22: 
                    grp = '13-22'
                    df = self.df3
                    
                # TRAIN DATA
                train_x = df.iloc[train_index]
                train_users = train_x.index.values
                train_y = self.df_labels.loc[self.df_labels.q==t].set_index('session').loc[train_users]
                
                # VALID DATA
                valid_x = df.iloc[test_index]
                valid_users = valid_x.index.values
                valid_y = self.df_labels.loc[self.df_labels.q==t].set_index('session').loc[valid_users]
                
                FEATURES = [c for c in df.columns if c != 'level_group']

                # TRAIN MODEL
                clf = RandomForestClassifier() 
                clf.fit(train_x[FEATURES], train_y['correct'])
                
                # SAVE MODEL, PREDICT VALID OOF
                models[f'{grp}_{t}'] = clf
                self.oof.loc[valid_users, t-1] = clf.predict_proba(valid_x[FEATURES])[:,1]

    def evaluate_validation(self, ):
        logger.info('Start evaluating validations.')

        # PUT TRUE LABELS INTO DATAFRAME WITH 18 COLUMNS
        true = self.oof.copy()
        for k in range(18):
            # GET TRUE LABELS
            tmp = self.df_labels.loc[self.df_labels.q == k+1].set_index('session').loc[self.ALL_USERS]
            true[k] = tmp.correct.values

        # FIND BEST THRESHOLD TO CONVERT PROBS INTO 1s AND 0s
        scores = []; thresholds = []
        best_score = 0; best_threshold = 0

        logger.info('Search optimal threshold.')
        for threshold in np.arange(0.4,0.81,0.01):
            preds = (self.oof.values.reshape((-1))>threshold).astype('int')
            m = f1_score(true.values.reshape((-1)), preds, average='macro')   
            scores.append(m)
            thresholds.append(threshold)
            if m>best_score:
                best_score = m
                best_threshold = threshold
        logger.info(f'optimal threshold: {best_threshold}')
        
        logger.info('When using optimal threshold...')
        self.scores = []
        
        m = f1_score(true.values.reshape((-1)), (self.oof.values.reshape((-1))>best_threshold).astype('int'), average='macro')
        logger.info(f'Overall F1 = {m}')
        scores.append(m)

        for k in range(18):
            m = f1_score(true[k].values, (self.oof[k].values>best_threshold).astype('int'), average='macro')
            logger.info(f'Q{k}: F1 = {m}')
            scores.append(m)
        
        logger.result_scores('test', scores)


    def write_sheet(self, ):

        data = [str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), self.run_fold_name] + self.scores
        data.append(self.load_options)
        data.append(self.validation_options)

        google_sheet = utils.WriteSheet()
        google_sheet.write(data, sheet_name='cv_scores')


    def main(self, ):

        self.load_dataset()
        self.engineer_features()
        self.run_validation()
        self.evaluate_validation()
        self.write_sheet()
        