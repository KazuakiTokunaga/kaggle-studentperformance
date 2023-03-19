import numpy as np
import pandas as pd
import polars as pl

from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from codes import utils, loader, preprocess


class Runner():

    def __init__(self, input_path='/kaggle/input/studentperformance-my/'):

        self.input_path = input_path


    def load_dataset(self, ):
        dataloader = loader.DataLoader(input_path=self.input_path)
        self.df_train, self.df_test, self.df_labels, self.df_submission = dataloader.load()
    

    def engineer_features(self, ):

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
        print('df1 done',df1.shape)
        self.df2 = preprocess.feature_engineer_pl(df2, grp='5-12', use_extra=True, feature_suffix='')
        print('df2 done',df2.shape)
        self.df3 = preprocess.feature_engineer_pl(df3, grp='13-22', use_extra=True, feature_suffix='')
        print('df3 done',df3.shape)
    

    def run_validation(self, ):

        self.df1 = utils.pl_to_pd(self.df1)
        self.df2 = utils.pl_to_pd(self.df2)
        self.df3 = utils.pl_to_pd(self.df3)

        ALL_USERS = self.df1.index.unique()
        print('We will train with', len(ALL_USERS) ,'users info')

        gkf = GroupKFold(n_splits=5)
        self.oof = pd.DataFrame(data=np.zeros((len(ALL_USERS),18)), index=ALL_USERS)
        models = {}

        # COMPUTE CV SCORE WITH 5 GROUP K FOLD
        for i, (train_index, test_index) in enumerate(gkf.split(X=self.df1, groups=self.df1.index)):

            # ITERATE THRU QUESTIONS 1 THRU 18
            for t in range(1,19):
                print(t,', ',end='')
                
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
                clf.fit(train_x[FEATURES].astype('float32'), train_y['correct'])
                
                # SAVE MODEL, PREDICT VALID OOF
                models[f'{grp}_{t}'] = clf
                self.oof.loc[valid_users, t-1] = clf.predict_proba(valid_x[FEATURES].astype('float32'))[:,1]

    def evaluate_validation(self, ):

        # PUT TRUE LABELS INTO DATAFRAME WITH 18 COLUMNS
        true = self.oof.copy()
        for k in range(18):
            # GET TRUE LABELS
            tmp = self.df_labels.loc[self.df_labels.q == k+1].set_index('session').loc[ALL_USERS]
            true[k] = tmp.correct.values

        # FIND BEST THRESHOLD TO CONVERT PROBS INTO 1s AND 0s
        scores = []; thresholds = []
        best_score = 0; best_threshold = 0

        for threshold in np.arange(0.4,0.81,0.01):
            print(f'{threshold:.02f}, ',end='')
            preds = (self.oof.values.reshape((-1))>threshold).astype('int')
            m = f1_score(true.values.reshape((-1)), preds, average='macro')   
            scores.append(m)
            thresholds.append(threshold)
            if m>best_score:
                best_score = m
                best_threshold = threshold

        print('When using optimal threshold...')
        for k in range(18):
                
            # COMPUTE F1 SCORE PER QUESTION
            m = f1_score(true[k].values, (self.oof[k].values>best_threshold).astype('int'), average='macro')
            print(f'Q{k}: F1 =',m)
            
        # COMPUTE F1 SCORE OVERALL
        m = f1_score(true.values.reshape((-1)), (self.oof.values.reshape((-1))>best_threshold).astype('int'), average='macro')
        print('==> Overall F1 =',m)


    def main(self, ):

        self.load_dataset()
        self.engineer_features()
        self.run_validation()
        self.evaluate_validation()
        