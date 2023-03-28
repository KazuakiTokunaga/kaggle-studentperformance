import numpy as np
import pandas as pd
import polars as pl
import gc
import json
import datetime

from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from lightgbm import LGBMClassifier
from lightgbm import early_stopping
from lightgbm import log_evaluation

from codes import utils, loader, preprocess

logger = utils.Logger()


class Runner():

    def __init__(self, 
        run_fold_name = 'run',
        repo_commit_hash = None,
        input_path='/kaggle/input/student-performance-my',
        repo_path='/kaggle/working/kaggle_studentperformance',
        load_options={
            'sampling': 1000,
            'split_labels': True,
            'parquet': True
        },
        validation_options={
            'n_fold': 2,
            'questions': list(range(1, 19))
        },
        model_options={
            'ensemble': False,
            'model': 'xgb',
            'param_file': 'params_xgb001_test.json'
        }):

        self.run_fold_name = run_fold_name
        self.repo_commit_hash = repo_commit_hash
        self.input_path = input_path
        self.repo_path = repo_path
        self.load_options = load_options
        self.model_options = model_options
        
        self.validation_options = validation_options
        self.n_fold = validation_options.get('n_fold')
        self.questions = validation_options.get('questions')


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
    

    def run_validation(self, 
            save_oof=True, 
            adhoc_params=None
        ):

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

        models = {}
        self.oof = pd.DataFrame(data=np.zeros((len(self.ALL_USERS),18)), index=self.ALL_USERS)
        self.best_ntrees = np.zeros([self.n_fold, 18])

        logger.info(f'Start validation with {self.n_fold} folds.')
        gkf = GroupKFold(n_splits=self.n_fold)
        for i, (train_index, test_index) in enumerate(gkf.split(X=self.df1, groups=self.df1.index)):

            # ITERATE THRU QUESTIONS 1 THRU 18
            for t in self.questions:
                if t <= 3:
                    logger.info(f'Fold {i} - Q {t}')
                
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
                if self.model_options.get('model') == 'xgb':
                    
                    param_file = self.model_options.get('param_file')
                    with open(f'{self.repo_path}/config/{param_file}') as f:
                        params = json.load(f)
                    
                    xgb_params = params['base']
    
                    n_estimators_list = params['n_estimators']
                    if len(n_estimators_list)==1:
                        xgb_params['n_estimators'] = n_estimators_list[0]
                    else:
                        xgb_params['n_estimators'] = n_estimators_list[t-1]

                    if adhoc_params:
                        for key, value in adhoc_params.items():
                            xgb_params[key] = value

                    if xgb_params.get('early_stopping_rounds'):
                        eval_set = [(valid_x[FEATURES], valid_y['correct'])]
                        clf = XGBClassifier(**xgb_params)
                        clf.fit(train_x[FEATURES], train_y['correct'], verbose = 0, eval_set=eval_set)
                        self.best_ntrees[i, t-1] = clf.best_ntree_limit
                    else:
                        clf =  XGBClassifier(**xgb_params)
                        clf.fit(train_x[FEATURES], train_y['correct'], verbose = 0)
                
                else:
                    clf = RandomForestClassifier() 
                    clf.fit(train_x[FEATURES], train_y['correct'])
                
                # SAVE MODEL, PREDICT VALID OOF
                models[f'{grp}_{t}'] = clf
                self.oof.loc[valid_users, t-1] = clf.predict_proba(valid_x[FEATURES])[:,1]

        if save_oof:
            self.oof.to_csv('oof_predict_proba.csv')

    def evaluate_validation(self, ):
        logger.info('Start evaluating validations.')

        # PUT TRUE LABELS INTO DATAFRAME WITH 18 COLUMNS
        true = self.oof.copy()
        for k in range(18):
            # GET TRUE LABELS
            tmp = self.df_labels.loc[self.df_labels.q == k+1].set_index('session').loc[self.ALL_USERS]
            true[k] = tmp.correct.values

        # Extract target columns.
        questions_idx = [i-1 for i in self.questions]
        oof_target = self.oof[question_idx].copy()
        true = true[question_idx]
        
        # FIND BEST THRESHOLD TO CONVERT PROBS INTO 1s AND 0s
        scores = []; thresholds = []
        best_score = 0; best_threshold = 0

        logger.info('Search optimal threshold.')
        for threshold in np.arange(0.45,0.75,0.01):
            preds = (oof_target.values.reshape((-1))>threshold).astype('int')
            m = f1_score(true.values.reshape((-1)), preds, average='macro')   
            scores.append(m)
            thresholds.append(threshold)
            if m>best_score:
                best_score = m
                best_threshold = threshold
        logger.info(f'optimal threshold: {best_threshold}')
        
        logger.info('When using optimal threshold...')
        self.scores = []
        
        m = f1_score(true.values.reshape((-1)), (oof_target.values.reshape((-1))>best_threshold).astype('int'), average='macro')
        logger.info(f'Overall F1 = {m}')
        self.scores.append(m)

        for k in range(18):
            m = f1_score(true[k].values, (oof_target[k].values>best_threshold).astype('int'), average='macro')
            logger.info(f'Q{k}: F1 = {m}')
            self.scores.append(m)
        
        if self.best_ntrees[0, 0] > 1:
            pd.Series(self.best_ntrees.mean(axis=0)).to_csv('best_num_trees.csv')


    def write_sheet(self, ):
        logger.info('Write scores to google sheet.')

        nowstr_jst = str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))
        data = [nowstr_jst, self.run_fold_name, self.repo_commit_hash] + self.scores
        data.append(self.load_options)
        data.append(self.validation_options)
        data.append(self.model_options)

        google_sheet = utils.WriteSheet()
        google_sheet.write(data, sheet_name='cv_scores')


    def main(self, ):

        self.load_dataset()
        self.engineer_features()
        self.run_validation()
        self.evaluate_validation()
        self.write_sheet()
        