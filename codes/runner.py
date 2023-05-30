import numpy as np
import pandas as pd
import polars as pl
import gc
import json
import os
import datetime
import pickle

import xgboost as xgb
import lightgbm as lgb
# from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from codes import utils, loader, preprocess


class Runner():

    def __init__(self, 
        run_fold_name = 'run',
        repo_commit_hash = None,
        input_path='/kaggle/input/student-performance-my',
        repo_path='/kaggle/working/kaggle_studentperformance',
        output_path='',
        log_path='',
        load_options={
            'sampling': 1000,
            'split_labels': True,
            'low_mem': False,
            'load_additional': False,
            'additional_grp2_3': False,
            'exclude_low_index': False,
            'exclude_coord_null': False
        },
        feature_options={
            'version': 2,
            'merge': False,
            'load_oof': False,
            'oof_file_name': '',
            'select': True,
            'exclude_suffix': '',
            'thre': 0.97,
            'time_id': 6,
            'level_diff': False,
            'cut_above': False,
            'room_click': False,
            'modify_elapsed_time': False,
            'use_csv': False,
            'add_random': False
        },
        validation_options={
            'n_fold': 2,
            'questions': list(range(1, 19)),
            'random_state': 42
        },
        model_options={
            'ensemble': False,
            'model': 'xgb',
            'param_file': 'params_xgb001_test.json',
            'use_gpu': False,
            'each_params': False,
            'each_folder_name': None, 
            'best_ntrees': [],
            'sample_weight': False,
            'weight_value': [],
            'random': True,
            'random_state': 42
        }):

        self.log_path = log_path
        self.output_path = output_path
        self.logger = utils.Logger(log_path)
        self.run_fold_name = run_fold_name
        self.repo_commit_hash = repo_commit_hash
        self.input_path = input_path
        self.repo_path = repo_path
        self.load_options = load_options
        self.feature_options = feature_options
        self.model_options = model_options
        self.models = {
            'features': {},
            'features_original': {},
            'models': {},
            'optimal_threshold': 0.620
        }
        self.model_kind = self.model_options.get('model')
        self.weight_value = self.model_options.get('weight_value')
        self.best_ntrees = model_options.get('best_ntrees')
        self.note = dict()
        self.fold_models_base = dict()
        self.fold_models = dict()
        
        self.validation_options = validation_options
        self.n_fold = validation_options.get('n_fold')
        self.questions = validation_options.get('questions')      


    def load_dataset(self, ):
        
        dataloader = loader.DataLoader(input_path=self.input_path, log_path=self.log_path, options=self.load_options)
        self.df_train, self.df_test, self.df_labels, self.df_submission = dataloader.load()

        if self.load_options.get('exclude_coord_null'):
            self.logger.info('Exclude coord null session_id in additional data.')
            exclude_session = pl.read_csv(f'{self.repo_path}/config/exclude_additional_session.csv').get_column('session_id').to_list()
            self.df_train = self.df_train.filter(~pl.col('session_id').is_in(exclude_session))

        if self.load_options.get('load_additional'):
            self.ORIGINAL_USERS = self.df_train.filter(pl.col('original')==1).get_column('session_id').unique()
            self.logger.info(f'Count of Original Users: {len(self.ORIGINAL_USERS)}')


    def delete_df_train(self, ):
        self.logger.info('Delete df_train and run a full collection.')

        del self.df_train
        gc.collect()
    

    def engineer_features(self, return_pd=True, fillna=True, save_model_params=False):
        self.logger.info('Start engineer features.')

        self.thre = self.feature_options.get('thre')
        self.logger.info(f'Threshold of null values: {self.thre}')

        self.add_random = self.feature_options.get('add_random')
        if self.add_random:
            self.logger.info('Add random features.')

        self.select = self.feature_options.get('select')
        self.exclude_suffix = self.feature_options.get('exclude_suffix')
        if self.select:
            self.logger.info(f'Select Features with suffix {self.exclude_suffix}.')
        self.merge_features = self.feature_options.get('merge')
        if self.merge_features:
            self.logger.info('Execute merge_features.')
        self.time_id = self.feature_options.get('time_id')
        if self.feature_options.get('level_diff'):
            self.logger.info('Add features based on elapsed_time level_diff.')
        if self.feature_options.get('cut_above'):
            self.logger.info('Extract big elapsed_time data to create additional features.')
        if self.feature_options.get('room_click'):
            self.logger.info('Add features based on navigate_click in each room.')

        self.df_train = preprocess.add_columns(self.df_train)

        params = {
            'use_extra': True,
            'version': self.feature_options.get('version'),
            'thre': 1-self.thre,
            'cut_above': self.feature_options.get('cut_above'),
            'level_diff': self.feature_options.get('level_diff'),
            'room_click': self.feature_options.get('room_click'),
            'use_csv': self.feature_options.get('use_csv'),
            'modify_elapsed_time': self.feature_options.get('modify_elapsed_time')
        }

        # sessionごとにまとめる
        grp = '0-4'
        self.df1 = preprocess.feature_engineer_pl(
            self.df_train.filter(pl.col("level_group")=='0-4'),
             grp=grp, 
             feature_suffix='grp0-4', 
             **params
        )
        self.df1 = preprocess.drop_columns(self.df1, thre=self.thre)
        self.df1 = preprocess.add_columns_session(self.df1, id=self.time_id)

        if self.add_random:
            self.df1 = preprocess.add_random_feature(self.df1)

        self.logger.info('df1 done.')

        grp = '5-12'
        self.df2 = preprocess.feature_engineer_pl(
            self.df_train.filter(pl.col("level_group")=='5-12'), 
            grp=grp, 
            feature_suffix='grp5-12',  
            **params
        )
        self.df2 = preprocess.drop_columns(self.df2, thre=self.thre)

        if self.merge_features:
            self.df2 = self.df2.join(self.df1, on='session_id', how='left')
        else:
            self.df2 = preprocess.add_columns_session(self.df2, id=self.time_id)
            if self.add_random:
                self.df2 = preprocess.add_random_feature(self.df2)

        self.logger.info('df2 done.')

        grp = '13-22'
        self.df3 = preprocess.feature_engineer_pl(
            self.df_train.filter(pl.col("level_group")=='13-22'),
            grp=grp,
            feature_suffix='grp13-22',
            **params)
        self.df3 = preprocess.drop_columns(self.df3, thre=self.thre)


        if self.merge_features:
            self.df3 = self.df3.join(self.df2, on='session_id', how='left')
        else:
            self.df3 = preprocess.add_columns_session(self.df3, id=self.time_id)
            if self.add_random:
                self.df3 = preprocess.add_random_feature(self.df3)

        self.logger.info('df3 done.')

        # 特徴量選択する前のカラム
        self.models['features_original']['0-4'] = self.df1.columns
        self.models['features_original']['5-12'] = self.df2.columns
        self.models['features_original']['13-22'] = self.df3.columns

        if self.select:
            self.logger.info('Execute feature selection.')
            exclude_df1_base = json.load(open(f'{self.repo_path}/config/exclude_df1{self.exclude_suffix}.json', 'r'))
            exclude_df2_base = json.load(open(f'{self.repo_path}/config/exclude_df2{self.exclude_suffix}.json', 'r'))
            exclude_df3_base = json.load(open(f'{self.repo_path}/config/exclude_df3{self.exclude_suffix}.json', 'r'))

            exclude_df1 = [i for i in exclude_df1_base if i in self.df1.columns]
            self.df1 = self.df1.drop(exclude_df1)

            exclude_df2 = [i for i in exclude_df2_base if i in self.df2.columns]
            self.df2 = self.df2.drop(exclude_df2)

            exclude_df3 = [i for i in exclude_df3_base if i in self.df3.columns]
            self.df3 = self.df3.drop(exclude_df3)


        self.models['features']['0-4'] = self.df1.columns
        self.models['features']['5-12'] = self.df2.columns
        self.models['features']['13-22'] = self.df3.columns

        self.logger.info(f'df1 shape: {self.df1.shape}')
        self.logger.info(f'df2 shape: {self.df2.shape}')
        self.logger.info(f'df3 shape: {self.df3.shape}')

        self.note['df1_shape'] = self.df1.shape
        self.note['df2_shape'] = self.df2.shape
        self.note['df3_shape'] = self.df3.shape

        if save_model_params:
            param_file = self.model_options.get('param_file')
            with open(f'{self.repo_path}/config/{param_file}') as f:
                params = json.load(f)
            self.note['model_params'] = params

        if return_pd:
            if type(self.df1) == pl.DataFrame:
                self.logger.info('Convert polars df to pandas df.')
                self.df1 = utils.pl_to_pd(self.df1)
                self.df2 = utils.pl_to_pd(self.df2)
                self.df3 = utils.pl_to_pd(self.df3)
            
            self.ALL_USERS = self.df1.index.unique()
            self.USERS2 = self.df2.index.unique()
            self.USERS3 = self.df3.index.unique()
        
            if fillna:
                self.logger.info('Execute fillna with -1 to pandas df.')
                self.df1 = self.df1.fillna(-1)
                self.df2 = self.df2.fillna(-1)
                self.df3 = self.df3.fillna(-1)
        

        if self.feature_options.get('load_oof'):
            self.file_name = self.feature_options.get('oof_file_name')
            self.logger.info(f'Load oof from csv.: {self.file_name}')
            
            self.oof = pd.read_csv(f'{self.input_path}/{self.file_name}.csv', index_col='session_id')
            self.oof.columns = [int(i) for i in self.oof.columns]

    def get_weight(self, df):

        idx_list = df.index

        res = []
        for id in idx_list:
            year = int(str(id)[:2])
            if year == 20:
                res.append(self.weight_value[0])
            elif year == 21:
                res.append(self.weight_value[1])
            elif year == 22:
                res.append(self.weight_value[2])
            else:
                res.append(1)

        return res

    def get_trained_clf(self, 
            t, 
            train_x, 
            train_y, 
            valid_x=None, 
            valid_y=None, 
            adhoc_params=None, 
            print_model_info=False,
            second = False
        ):
            
        validation = valid_x is not None
        ntree = None
        model_kind = self.model_options.get('model')
        param_file = self.model_options.get('param_file')

        with open(f'{self.repo_path}/config/{param_file}') as f:
            params = json.load(f)
    
        model_params = params['base']

        # Qごとにパラメタを変更する場合
        if self.model_options.get('each_params'):
            folder_name = self.model_options.get('each_folder_name')
            filepath = f'{self.repo_path}/config/{folder_name}/q{t}.json'
            
            if os.path.isfile(filepath):

                if print_model_info:
                    self.logger.info('Use customized paramter for this question.')

                with open(filepath) as f:
                    params = json.load(f)
                model_params = params['base']

        # Qごとにn_estimatorsを変えるかどうか
        n_estimators_list = params['n_estimators']
        if len(n_estimators_list)==1:
            model_params['n_estimators'] = n_estimators_list[0]
        else:
            model_params['n_estimators'] = n_estimators_list[t-1]

        if self.model_options.get('use_gpu') and model_kind == 'xgb':
            if print_model_info:
                self.logger.info('Use GPU.')
            model_params['tree_method'] = 'gpu_hist'

        if self.model_options.get('random'):
            model_params['random_state'] = np.random.randint(1, 100)
        elif self.model_options.get('random_state'):
            model_params['random_state'] = self.model_options.get('random_state')

        # validation時にbest_iterationを保存している場合はそちらを優先する
        if len(self.best_ntrees) > 0:
            n = self.best_ntrees[t-1]
            self.logger.info(f'Q{t}: n_estimators {n}')
            model_params['n_estimators'] = n

        if adhoc_params:
            for key, value in adhoc_params.items():
                model_params[key] = value

        if print_model_info:
            self.logger.info(f'Use {model_kind} with params {model_params}.')

        # early_stoppingを用いる場合
        if validation and model_params.get('early_stopping_rounds'):
            if print_model_info:
                self.logger.info(f'Use early_stopping_rounds.')

            eval_set = [(valid_x, valid_y['correct'])]
            
            if model_kind == 'xgb':
                clf = xgb.XGBClassifier(**model_params)
                if self.model_options.get('sample_weight'):
                    if print_model_info:
                        self.logger.info(f'Use sampling weight.')
                    clf.fit(
                        train_x, 
                        train_y['correct'],
                        sample_weight = self.get_weight(train_x), 
                        verbose = 0, 
                        eval_set=eval_set,
                        sample_weight_eval_set=[self.get_weight(valid_x)]
                    )
                else:
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

                if self.model_options.get('sample_weight'):
                    if print_model_info:
                        self.logger.info(f'Use sampling weight.')
                    clf.fit(
                        train_x, 
                        train_y['correct'],
                        sample_weight = self.get_weight(train_x), 
                        verbose = 0, 
                    )
                else:
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

        return clf, ntree

    def run_validation(self, 
            save_oof=True, 
            adhoc_params=None,
            save_fold_models=True,
            adhoc_questions=None
        ):

        user_cnt = len(self.ALL_USERS)
        self.logger.info(f'We will train with {user_cnt} users info')

        arr = [0.728, 0.978, 0.933, 0.8, 0.548, 0.776, 0.736, 0.612, 0.734, 0.505, 0.642, 0.86 , 0.275, 0.707, 0.481, 0.733, 0.684, 0.95]
        if not self.feature_options.get('load_oof'):
            self.oof = pd.DataFrame(data=np.multiply(np.ones((len(self.ALL_USERS), 1)), arr), index=self.ALL_USERS) # Question t はカラム t-1 に対応する
        best_ntrees_mat = np.zeros([self.n_fold, 18])

        random_state_validation = self.validation_options.get('random_state')
        self.logger.info(f'Start validation with {self.n_fold} folds, random_state {random_state_validation}.')
        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state = random_state_validation)
        kf_split_list = list(kf.split(X=self.df1))

        # ハイパラ最適化のため
        if adhoc_questions is not None:
            self.questions = adhoc_questions

        for t in self.questions:

            for k, (train_index, test_index) in enumerate(kf_split_list):
                print_model_info = False if k else True

                if k==0 or (t <= 2 and k <= 2):
                    self.logger.info(f'Question {t}, Fold {k}.')
                
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
                prev_answers = self.oof.loc[train_users, [i for i in range(t-1)]].copy()
                train_x = train_x.merge(prev_answers, left_index=True, right_index=True, how='left')
                train_y = self.df_labels.loc[self.df_labels.q==t].set_index('session').loc[train_users]

                
                # VALID DATA
                valid_x = df.iloc[test_index]
                valid_users = valid_x.index.values
                prev_answers = self.oof.loc[valid_users, [i for i in range(t-1)]].copy()
                valid_x = valid_x.merge(prev_answers, left_index=True, right_index=True, how='left')
                valid_y = self.df_labels.loc[self.df_labels.q==t].set_index('session').loc[valid_users]

                clf, ntree = self.get_trained_clf(t, train_x, train_y, valid_x, valid_y, adhoc_params, print_model_info=print_model_info)
                best_ntrees_mat[k, t-1] = ntree
                
                self.oof.loc[valid_users, t-1] = clf.predict_proba(valid_x)[:,1]

                if save_fold_models:
                    self.fold_models[f'q{t}_fold{k}'] = clf

        if best_ntrees_mat[0, 0] > 1:
            self.logger.info('Save best iterations.')
            best_ntrees = pd.Series(best_ntrees_mat.mean(axis=0).astype('int'))
            best_ntrees.to_csv(f'{self.output_path}best_num_trees.csv')
            self.best_ntrees = list(best_ntrees)
            self.note['best_ntrees'] = self.best_ntrees
            

        if save_oof:
            self.logger.info('Export oof_predict_proba.')
            self.oof.to_csv(f'{self.output_path}oof_predict_proba.csv')
        
        if save_fold_models:
            pickle.dump(self.fold_models, open(f'{self.output_path}fold_models.pkl', 'wb'))


    def run_validation_with_addition(self, 
            save_oof=True, 
            adhoc_params=None,
            save_fold_models=True,
            adhoc_questions=None
        ):

        user_cnt = len(self.ALL_USERS)
        self.logger.info(f'We will train with {user_cnt} users info')

        arr = [0.728, 0.978, 0.933, 0.8, 0.548, 0.776, 0.736, 0.612, 0.734, 0.505, 0.642, 0.86 , 0.275, 0.707, 0.481, 0.733, 0.684, 0.95]
        if not self.feature_options.get('load_oof'):
            # self.oof = pd.DataFrame(data=np.multiply(np.ones((len(self.ALL_USERS), 1)), arr), index=self.ALL_USERS) # Question t はカラム t-1 に対応する
            self.oof = pd.DataFrame(data=np.zeros((len(self.ALL_USERS), 18)), index=self.ALL_USERS) # Question t はカラム t-1 に対応する
            self.logger.info(f'shape oof: {self.oof.shape}')
        best_ntrees_mat = np.zeros([self.n_fold, 18])

        random_state_validation = self.validation_options.get('random_state')
        self.logger.info(f'Start validation with {self.n_fold} folds, random_state {random_state_validation}.')
        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state = random_state_validation)
        
        kf_split_list1 = list(kf.split(X=self.df1))
        kf_split_list2 = list(kf.split(X=self.df2))
        kf_split_list3 = list(kf.split(X=self.df3))

        # ハイパラ最適化のため
        if adhoc_questions is not None:
            self.questions = adhoc_questions

        for t in self.questions:
    
            if t<=3: 
                grp = '0-4'
                df = self.df1
                kf_split_list = kf_split_list1
            elif t<=13: 
                grp = '5-12'
                df = self.df2
                kf_split_list = kf_split_list2
            elif t<=22: 
                grp = '13-22'
                df = self.df3
                kf_split_list = kf_split_list3

            for k, (train_index, test_index) in enumerate(kf_split_list):
                print_model_info = False if k else True

                if k==0 or (t <= 2 and k <= 2):
                    self.logger.info(f'Question {t}, Fold {k}.')
                
                # TRAIN DATA
                train_x = df.iloc[train_index]
                train_users = train_x.index.values
                prev_answers = self.oof.loc[train_users, [i for i in range(t-1)]].copy()
                train_x = train_x.merge(prev_answers, left_index=True, right_index=True, how='left')
                train_y = self.df_labels.loc[self.df_labels.q==t].set_index('session').loc[train_users]

                
                # VALID DATA
                valid_x = df.iloc[test_index]
                valid_users = valid_x.index.values
                prev_answers = self.oof.loc[valid_users, [i for i in range(t-1)]].copy()
                valid_x = valid_x.merge(prev_answers, left_index=True, right_index=True, how='left')
                valid_y = self.df_labels.loc[self.df_labels.q==t].set_index('session').loc[valid_users]

                clf, ntree = self.get_trained_clf(t, train_x, train_y, valid_x, valid_y, adhoc_params, print_model_info=print_model_info)
                best_ntrees_mat[k, t-1] = ntree
                
                self.oof.loc[valid_users, t-1] = clf.predict_proba(valid_x)[:,1]

                if save_fold_models:
                    self.fold_models[f'q{t}_fold{k}'] = clf

        if best_ntrees_mat[0, 0] > 1:
            self.logger.info('Save best iterations.')
            best_ntrees = pd.Series(best_ntrees_mat.mean(axis=0).astype('int'))
            best_ntrees.to_csv(f'{self.output_path}best_num_trees.csv')
            self.best_ntrees = list(best_ntrees)
            self.note['best_ntrees'] = self.best_ntrees
            

        if save_oof:
            self.logger.info('Export oof_predict_proba.')
            self.oof.to_csv(f'{self.output_path}oof_predict_proba.csv')
        
        if save_fold_models:
            pickle.dump(self.fold_models, open(f'{self.output_path}fold_models.pkl', 'wb'))


    def evaluate_validation(self, ):
        self.logger.info('Start evaluating validations.')

        # PUT TRUE LABELS INTO DATAFRAME WITH 18 COLUMNS
        true = pd.DataFrame(data=np.zeros((len(self.ALL_USERS),18)), index=self.ALL_USERS)
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

        self.logger.info('Search optimal threshold.')
        for threshold in np.arange(0.60,0.65,0.001):
            preds = (oof_target.values.reshape((-1))>threshold).astype('int')
            m = f1_score(true.values.reshape((-1)), preds, average='macro')   
            scores.append(m)
            thresholds.append(threshold)
            if m>best_score:
                best_score = m
                best_threshold = threshold

        self.models['optimal_threshold'] = np.round(best_threshold, 6)
        self.note['best_threshold'] = best_threshold
        self.logger.info(f'optimal threshold: {best_threshold}')
        
        self.logger.info('When using optimal threshold...')
        self.scores = []
        
        m = f1_score(true.values.reshape((-1)), (oof_target.values.reshape((-1))>best_threshold).astype('int'), average='macro')
        self.logger.info(f'Overall F1 = {m}')
        self.scores.append(m)

        for k in question_idx:
            m = f1_score(true[k].values, (oof_target[k].values>best_threshold).astype('int'), average='macro')
            self.logger.info(f'Q{k}: F1 = {m}')
            self.scores.append(m)



    def evaluate_validation_with_addition(self, ):
        self.logger.info('Start evaluating validations.')

        # PUT TRUE LABELS INTO DATAFRAME WITH 18 COLUMNS
        true = pd.DataFrame(data=np.zeros((len(self.ORIGINAL_USERS),18)), index=self.ORIGINAL_USERS)
        for k in range(18):
            # GET TRUE LABELS
            tmp = self.df_labels.loc[self.df_labels.q == k+1].set_index('session').loc[self.ORIGINAL_USERS]
            true[k] = tmp.correct.values

        # Extract target columns.
        question_idx = [i-1 for i in self.questions]
        oof_target = self.oof.loc[self.ORIGINAL_USERS, question_idx].copy()
        true = true[question_idx]

        self.logger.info(f'Evaluate oof shape: {oof_target.shape}')
        
        # FIND BEST THRESHOLD TO CONVERT PROBS INTO 1s AND 0s
        scores = []; thresholds = []
        best_score = 0; best_threshold = 0

        self.logger.info('Search optimal threshold.')
        for threshold in np.arange(0.60,0.65,0.001):
            preds = (oof_target.values.reshape((-1))>threshold).astype('int')
            m = f1_score(true.values.reshape((-1)), preds, average='macro')   
            scores.append(m)
            thresholds.append(threshold)
            if m>best_score:
                best_score = m
                best_threshold = threshold

        self.models['optimal_threshold'] = np.round(best_threshold, 6)
        self.note['best_threshold'] = best_threshold
        self.logger.info(f'optimal threshold: {best_threshold}')
        
        self.logger.info('When using optimal threshold...')
        self.scores = []
        
        m = f1_score(true.values.reshape((-1)), (oof_target.values.reshape((-1))>best_threshold).astype('int'), average='macro')
        self.logger.info(f'Overall F1 = {m}')
        self.scores.append(m)

        for k in question_idx:
            m = f1_score(true[k].values, (oof_target[k].values>best_threshold).astype('int'), average='macro')
            self.logger.info(f'Q{k}: F1 = {m}')
            self.scores.append(m)


    def train_all_clf(self, save_model=True):
        self.logger.info(f'Train clf using all train data.')

        # ITERATE THRU QUESTIONS 1 THRU 18
        for t in self.questions:
            self.logger.info(f'Question {t}.')
            
            # USE THIS TRAIN DATA WITH THESE QUESTIONS
            if t<=3: 
                grp = '0-4'
                df = self.df1
                USERS = self.ALL_USERS
            elif t<=13: 
                grp = '5-12'
                df = self.df2
                USERS = self.USERS2 if self.load_options.get('load_additional') else self.ALL_USERS
            elif t<=22: 
                grp = '13-22'
                df = self.df3
                USERS = self.USERS3 if self.load_options.get('load_additional') else self.ALL_USERS
                
            # TRAIN DATA
            train_x = df
            prev_answers = self.oof.loc[USERS, [i for i in range(t-1)]].copy()
            train_x = train_x.merge(prev_answers, left_index=True, right_index=True, how='left')
            
            train_y = self.df_labels.loc[self.df_labels.q==t].set_index('session').loc[USERS]

            clf, ntree = self.get_trained_clf(t, train_x, train_y, print_model_info=True)

            # SAVE MODEL.
            if self.model_kind == 'xgb':
                clf.save_model(f'{self.output_path}models/xgb_q{t}.json')

            self.models['models'][f'{grp}_{t}'] = clf

        self.logger.info(f'Saved trained model.')

        pickle.dump(self.models, open(f'{self.output_path}models.pkl', 'wb'))
        self.logger.info('Export trained model.')


    def write_sheet(self, ):
        self.logger.info('Write scores to google sheet.')

        nowstr_jst = str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))
        data = [nowstr_jst, self.run_fold_name, self.repo_commit_hash] + self.scores
        data.append(self.load_options)
        data.append(self.validation_options)
        data.append(self.model_options)
        data.append(self.feature_options)

        data.append(self.note)

        google_sheet = utils.WriteSheet(
            json_key = f'{self.input_path}/ktokunaga-4094cf694f5c.json'
        )
        google_sheet.write(data, sheet_name='cv_scores')


    def main(self, ):

        self.load_dataset()
        self.engineer_features()
        self.delete_df_train()
        self.run_validation()
        self.evaluate_validation()
        self.write_sheet()
        self.train_all_clf()
        