import numpy as np
import pandas as pd

def summarize_logs(df_train):
    CATS = ['event_name', 'name','fqid', 'room_fqid', 'text_fqid']
    NUMS = ['elapsed_time','level','page','room_coor_x', 'room_coor_y', 
            'screen_coor_x', 'screen_coor_y', 'hover_duration']

    dfs = []
    for c in CATS:
        tmp = df_train.groupby(['session_id','level_group'])[c].agg('nunique')
        tmp.name = tmp.name + '_nunique'
        dfs.append(tmp)
    for c in NUMS:
        tmp = df_train.groupby(['session_id','level_group'])[c].agg('mean')
        dfs.append(tmp)
    for c in NUMS:
        tmp = df_train.groupby(['session_id','level_group'])[c].agg('std')
        tmp.name = tmp.name + '_std'
        dfs.append(tmp)
    df = pd.concat(dfs,axis=1)
    df = df.fillna(-1)
    df = df.reset_index()
    df = df.set_index('session_id')
    return df