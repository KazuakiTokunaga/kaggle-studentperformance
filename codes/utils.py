import logging
import datetime
import gspread
import json
import os
import subprocess

import pandas as pd
import numpy as np
import polars as pl

from oauth2client.service_account import ServiceAccountCredentials


def pl_to_pd(df, index_col='session_id'):
    return df.to_pandas().set_index(index_col)


def get_commit_hash(repo_path='/kaggle/working/kaggle_studentperformance/'):

    wd = os.getcwd()
    os.chdir(repo_path)
    
    cmd = "git show --format='%H' --no-patch"
    hash_value = subprocess.check_output(cmd.split()).decode('utf-8')[1:-3]

    os.chdir(wd)

    return hash_value
    

class Logger:

    def __init__(self, log_path='/kaggle/working/'):
        self.general_logger = logging.getLogger('general')
        self.result_logger = logging.getLogger('result')
        stream_handler = logging.StreamHandler()
        file_general_handler = logging.FileHandler(f'{log_path}general.log')
        file_result_handler = logging.FileHandler(f'{log_path}result.log')
        if len(self.general_logger.handlers) == 0:
            self.general_logger.addHandler(stream_handler)
            self.general_logger.addHandler(file_general_handler)
            self.general_logger.setLevel(logging.INFO)
            self.result_logger.addHandler(stream_handler)
            self.result_logger.addHandler(file_result_handler)
            self.result_logger.setLevel(logging.INFO)

    def info(self, message):
        # 時刻をつけてコンソールとログに出力
        self.general_logger.info('[{}] - {}'.format(self.now_string(), message))

    def result(self, message):
        self.result_logger.info(message)

    def result_ltsv(self, dic):
        self.result(self.to_ltsv(dic))

    def result_scores(self, run_name, scores):
        # 計算結果をコンソールと計算結果用ログに出力
        dic = dict()
        dic['name'] = run_name
        dic['overall_score'] = scores[0]
        for i, score in enumerate(scores[1:]):
            dic[f'score{i}'] = score
        self.result(self.to_ltsv(dic))

    def now_string(self):
        return str(datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S'))

    def to_ltsv(self, dic):
        return '\t'.join(['{}:{}'.format(key, value) for key, value in dic.items()])


class WriteSheet:

    def __init__(self, 
        json_key='/kaggle/input/student-performance-my/ktokunaga-4094cf694f5c.json',
        sheet_key = '1NCWjO_3V99tLybvTiwxO54_bZVMpLB1G_IEo2fq7NVk',
    ):
        
        scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(json_key, scope)
        gc = gspread.authorize(credentials)
        self.worksheet = gc.open_by_key(sheet_key)
    

    def write(self, data, sheet_name, table_range='A1'):

        sheet = self.worksheet.worksheet(sheet_name)

        # 辞書のみJSONに変換、ほかはそのままにして、書き込む
        data_json = [json.dumps(d, ensure_ascii=False) if type(d) == dict else d for d in data]
        sheet.append_row(data_json, table_range=table_range)