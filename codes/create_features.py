import numpy as np
import pandas as pd
import polars as pl

from sklearn.preprocessing import StandardScaler
import umap
import pickle

def create_room_umap_model(df):

    df_navigate = df.filter((pl.col('event_name')=='navigate_click')&(pl.col('fqid')=='fqid_None'))
    df_navigate = df_navigate.with_columns([
        (pl.col('room_coor_x') // 30).cast(pl.Int64).alias('room_x'),
        (pl.col('room_coor_y') // 30).cast(pl.Int64).alias('room_y')
    ])
    print('df_navigate_shape:', df_navigate.shape)

    
    room_umap_model = {
        'sc': dict()
        'umap': dict(),
        'features': dict()
    }

    grps = ['0-4', '5-12', '13-22']
    
    for grp in grps:

        df_navigate_grp = df_navigate.filter(pl.col('level_group')==grp)
        
        room_umap_model['sc'][grp] = dict()
        room_umap_model['umap'][grp] = dict()
        room_umap_model['features'][grp] = dict()

        if grp == '0-4':
            rooms = ['tunic.kohlcenter.halloffame', 'tunic.historicalsociety.stacks', 'tunic.historicalsociety.basement', 'tunic.historicalsociety.collection', 'tunic.historicalsociety.entry', 'tunic.historicalsociety.closet']
        elif grp == '5-12':
            rooms = ['tunic.historicalsociety.frontdesk', 'tunic.capitol_0.hall', 'tunic.capitol_1.hall', 'tunic.library.microfiche', 'tunic.historicalsociety.closet_dirty', 'tunic.historicalsociety.basement', 'tunic.historicalsociety.collection', 'tunic.library.frontdesk', 'tunic.historicalsociety.stacks', 'tunic.drycleaner.frontdesk', 'tunic.historicalsociety.entry', 'tunic.humanecology.frontdesk', 'tunic.kohlcenter.halloffame']
        else:
            rooms = ['tunic.historicalsociety.stacks', 'tunic.flaghouse.entry', 'tunic.kohlcenter.halloffame', 'tunic.capitol_2.hall', 'tunic.historicalsociety.closet_dirty', 'tunic.historicalsociety.basement', 'tunic.capitol_1.hall', 'tunic.historicalsociety.entry', 'tunic.historicalsociety.frontdesk', 'tunic.library.frontdesk', 'tunic.historicalsociety.collection_flag', 'tunic.humanecology.frontdesk', 'tunic.wildlife.center', 'tunic.library.microfiche', 'tunic.historicalsociety.cage']
        
        for r in rooms:

            df_room = df_navigate_grp.filter(pl.col('room_fqid')==r)
            df_dummies = df_room.select('session_id', 'room_x', 'room_y').to_dummies(columns = ['room_x', 'room_y'])
            df_room_summary = df_dummies.groupby('session_id').sum().to_pandas().set_index('session_id').clip(0, 3)
            room_umap_model['features'][grp][r] = list(df_room_summary.columns)
            
            sc = StandardScaler()
            sc.fit(df_room_summary.to_numpy())
            room_umap_model['sc'][grp][r] = sc

            df_room_std= pd.DataFrame(data=sc.transform(df_room_summary.to_numpy()), columns=df_room_summary.columns, index=df_room_summary.index)
            um = umap.UMAP(random_state=2)
            um.fit(df_room_summary.to_numpy())
            room_umap_model['umap'][grp][r] = um
    

    pickle.dump(room_umap_model, open(f'room_umap_model.pkl', 'wb'))