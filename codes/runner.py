import numpy as np
import pandas as pd
import polars as pl
import gc
import json
import datetime
import pickle

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

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
        self.models = {
            'features': {},
            'models': {},
            'optimal_threshold': 0.620
        }
        self.print_model_info = True
        self.best_ntrees = None
        self.note = dict()
        
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
    

    def engineer_features(self, return_pd=True, fillna=True):
        logger.info('Start engineer features.')

        self.df_train = preprocess.add_columns(self.df_train)

        # グループごとに分割
        df1 = self.df_train.filter(pl.col("level_group")=='0-4')
        df2 = self.df_train.filter(pl.col("level_group")=='5-12')
        df3 = self.df_train.filter(pl.col("level_group")=='13-22')

        # sessionごとにまとめる
        grp = '0-4'
        self.df1 = preprocess.feature_engineer_pl(df1, grp=grp, use_extra=True, feature_suffix='')
        self.df1 = preprocess.drop_columns(self.df1)
        self.models['features'][grp] = self.df1.columns
        logger.info(f'df1 done: {self.df1.shape}')
        
        grp = '5-12'
        df2 = preprocess.feature_engineer_pl(df2, grp=grp, use_extra=True, feature_suffix='')
        self.df2 = preprocess.drop_columns(df2)
        self.models['features'][grp] = self.df2.columns
        logger.info(f'df2 done: {self.df2.shape}')

        grp = '13-22'
        df3 = preprocess.feature_engineer_pl(df3, grp=grp, use_extra=True, feature_suffix='')
        self.df3 = preprocess.drop_columns(df3)
        self.models['features'][grp] = self.df3.columns
        logger.info(f'df3 done: {self.df3.shape}')

        self.note['df1_shape'] = self.df1.shape
        self.note['df2_shape'] = self.df2.shape
        self.note['df3_shape'] = self.df3.shape
        self.note['feature'] = {
            "time": 'year,month,dayはなし',
            "ver2": '集約を追加、不審なクリックを追加'
        }

        if return_pd:
            if type(self.df1) == pl.DataFrame:
                logger.info('Convert polars df to pandas df.')
                self.df1 = utils.pl_to_pd(self.df1)
                self.df2 = utils.pl_to_pd(self.df2)
                self.df3 = utils.pl_to_pd(self.df3)
        
            if fillna:
                logger.info('Execute fillna with -1 to pandas df.')
                self.df1 = self.df1.fillna(-1)
                self.df2 = self.df2.fillna(-1)
                self.df3 = self.df3.fillna(-1)

    def get_trained_clf(self, t, train_x, train_y, valid_x=None, valid_y=None, adhoc_params=None):
            
        validation = valid_x is not None
        ntree = None
        model_kind = self.model_options.get('model')
        param_file = self.model_options.get('param_file')

        with open(f'{self.repo_path}/config/{param_file}') as f:
            params = json.load(f)
    
        model_params = params['base']

        # Qごとにn_estimatorsを変えるかどうか
        n_estimators_list = params['n_estimators']
        if len(n_estimators_list)==1:
            model_params['n_estimators'] = n_estimators_list[0]
        else:
            model_params['n_estimators'] = n_estimators_list[t-1]

        # validation時にbest_iterationを保存している場合はそちらを優先する
        if self.best_ntrees is not None:
            n = self.best_ntrees[t-1]
            logger.info(f'Q{t}: n_estimators {n}')
            model_params['n_estimators'] = n

        if adhoc_params:
            for key, value in adhoc_params.items():
                model_params[key] = value

        if self.print_model_info:
            logger.info(f'Use {model_kind} with params {model_params}.')

        # early_stoppingを用いる場合
        if validation and model_params.get('early_stopping_rounds'):
            if self.print_model_info:
                logger.info(f'Use early_stopping_rounds.')

            eval_set = [(valid_x, valid_y['correct'])]
            
            if model_kind == 'xgb':
                clf = xgb.XGBClassifier(**model_params)
                clf.fit(train_x, train_y['correct'], verbose = 0, eval_set=eval_set)
                ntree = clf.best_ntree_limit
            
            elif model_kind == 'lgb':
                stopping_rounds = model_params.pop('early_stopping_rounds')

                clf = lgb.LGBMClassifier(**model_params)
                clf.fit(
                    train_x, train_y['correct'], eval_set=eval_set, 
                    callbacks=[
                            lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False), # early_stopping用コールバック関数
                            lgb.log_evaluation(0)
                    ] 
                )
                ntree = clf.best_iteration_
            
            elif model_kind == 'cat':
                
                train_pool = Pool(train_x.astype('float32'), train_y['correct'])
                valid_pool = Pool(valid_x.astype('float32'), valid_y['correct'])

                clf = CatBoostClassifier(**model_params)
                clf.fit(train_pool, eval_set=valid_pool)

                ntree = clf.get_best_iteration()
            
            else:
                raise Exception('Wrong Model kind with early stopping.')

        # early stoppingを用いない場合
        else:
            
            # train全体で予測する場合、除外する必要
            if model_params.get('early_stopping_rounds'):
                model_params.pop('early_stopping_rounds')

            if model_kind == 'xgb':
                clf =  xgb.XGBClassifier(**model_params)
                clf.fit(train_x, train_y['correct'], verbose = 0)
            
            elif model_kind == 'lgb':
                clf = lgb.LGBMClassifier(**model_params)
                clf.fit(train_x, train_y['correct'], callbacks=[lgb.log_evaluation(0)])
            
            elif model_kind == 'cat':
                train_pool = Pool(train_x.astype('float32'), train_y['correct'])
                clf = CatBoostClassifier(**model_params)
                clf.fit(train_pool)
        
            elif model_kind == 'rf':
                clf = RandomForestClassifier(**model_params) 
                clf.fit(train_x, train_y['correct'])
            
            else:
                raise Exception('Wrong Model kind.')
        
        self.print_model_info = False

        return clf, ntree


    def run_validation(self, 
            save_oof=True, 
            adhoc_params=None
        ):

        self.ALL_USERS = self.df1.index.unique()
        user_cnt = len(self.ALL_USERS)
        logger.info(f'We will train with {user_cnt} users info')

        self.oof = pd.DataFrame(data=np.zeros((len(self.ALL_USERS),18)), index=self.ALL_USERS)
        best_ntrees_mat = np.zeros([self.n_fold, 18])

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
                prev_answers = self.oof.loc[train_users, [i for i in range(1, t)]].copy()
                train_x = train_x.merge(prev_answers, left_index=True, right_index=True, how='left')
                self.train_x = train_x

                train_y = self.df_labels.loc[self.df_labels.q==t].set_index('session').loc[train_users]
                
                # VALID DATA
                valid_x = df.iloc[test_index]
                valid_users = valid_x.index.values
                prev_answers = self.oof.loc[valid_users, [i for i in range(1, t)]].copy()
                valid_x = valid_x.merge(prev_answers, how='left')
                valid_y = self.df_labels.loc[self.df_labels.q==t].set_index('session').loc[valid_users]

                clf, ntree = self.get_trained_clf(t, train_x, train_y, valid_x, valid_y, adhoc_params)
                best_ntrees_mat[i, t-1] = ntree
                
                self.oof.loc[valid_users, t-1] = clf.predict_proba(valid_x)[:,1]

        if best_ntrees_mat[0, 0] > 1:
            logger.info('Save best iterations.')
            self.best_ntrees = pd.Series(best_ntrees_mat.mean(axis=0).astype('int'))
            self.best_ntrees.to_csv('best_num_trees.csv')
            self.note['best_ntrees'] = list(self.best_ntrees)
            

        if save_oof:
            logger.info('Export oof_predict_proba.')
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
        question_idx = [i-1 for i in self.questions]
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

        self.models['optimal_threshold'] = np.round(best_threshold, 6)
        self.note['best_threshold'] = best_threshold
        logger.info(f'optimal threshold: {best_threshold}')
        
        logger.info('When using optimal threshold...')
        self.scores = []
        
        m = f1_score(true.values.reshape((-1)), (oof_target.values.reshape((-1))>best_threshold).astype('int'), average='macro')
        logger.info(f'Overall F1 = {m}')
        self.scores.append(m)

        for k in question_idx:
            m = f1_score(true[k].values, (oof_target[k].values>best_threshold).astype('int'), average='macro')
            logger.info(f'Q{k}: F1 = {m}')
            self.scores.append(m)


    def train_all_clf(self, save_model=True):
        logger.info(f'Train clf using all train data.')

        # ITERATE THRU QUESTIONS 1 THRU 18
        for t in self.questions:
            
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
            train_x = df
            train_y = self.df_labels.loc[self.df_labels.q==t].set_index('session')

            clf, ntree = self.get_trained_clf(t, train_x, train_y)    

            # SAVE MODEL.
            self.models['models'][f'{grp}_{t}'] = clf

        logger.info(f'Saved trained model.')

        pickle.dump(self.models, open(f'models.pkl', 'wb'))
        logger.info('Export trained model.')


    def write_sheet(self, ):
        logger.info('Write scores to google sheet.')

        nowstr_jst = str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))
        data = [nowstr_jst, self.run_fold_name, self.repo_commit_hash] + self.scores
        data.append(self.load_options)
        data.append(self.validation_options)
        data.append(self.model_options)
        data.append(self.note)

        google_sheet = utils.WriteSheet()
        google_sheet.write(data, sheet_name='cv_scores')


    def main(self, ):

        self.load_dataset()
        self.engineer_features()
        self.delete_df_train()
        self.run_validation()
        self.evaluate_validation()
        self.write_sheet()
        self.train_all_clf()
        