import numpy as np
import pandas as pd
import polars as pl
import logging


def split_df_labels(df_labels):

    df_labels['session'] = df_labels.session_id.apply(lambda x: int(x.split('_')[0]) )
    df_labels['q'] = df_labels.session_id.apply(lambda x: int(x.split('_')[-1][1:]) )

    return df_labels


def add_columns(df):

    columns = [

        pl.col("page").cast(pl.Float32),
        (
            (pl.col("elapsed_time") - pl.col("elapsed_time").shift(1)) # 前のアクションから
            .fill_null(0)
            .clip(0, 1e9)
            .over(["session_id", "level_group"])
            .alias("elapsed_time_diff")
        ),
        (
            (pl.col("elapsed_time").shift(-1) - pl.col("elapsed_time")) # 次のアクションまで 
            .fill_null(0)
            .clip(0, 1e9)
            .over(["session_id", "level_group"])
            .alias("elapsed_time_diff_to")
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

    df = df.with_columns(columns)

    return df


def drop_columns(df, thre=0.97):

    null_rates_raw = df.null_count() / df.height
    null_rates = (null_rates_raw <= thre)
    columns_flag = null_rates.to_numpy()[0]
    columns = list(np.array(df.columns)[columns_flag])

    # aboveがつくカラムは例外(閾値超え)
    excep_null_rates = (null_rates_raw <= 0.997)
    excep_columns_flag = excep_null_rates.to_numpy()[0]
    excep_columns = list(np.array(df.columns)[excep_columns_flag])
    excep_columns = [i for i in excep_columns if i not in columns and i.endswith('_above')]

    df = df.select(columns + excep_columns)

    drop_columns = []
    for col in df.columns:
        if df.get_column(col).n_unique() == 1:
            drop_columns.append(col)
    
    df = df.drop(drop_columns)

    return df


def feature_engineer_pl(x, grp, 
        use_extra=True, 
        feature_suffix = '', 
        version=2, 
        use_csv=False, 
        csv_path='',
        thre = 0.03,
        level_diff=True,
        cut_above=True
    ):

    # from https://www.kaggle.com/code/leehomhuang/catboost-baseline-with-lots-features-inference :

    CATS = ['event_name', 'name', 'fqid', 'room_fqid', 'text_fqid']
    NUMS = ['page', 'room_coor_x', 'room_coor_y', 'screen_coor_x', 'screen_coor_y',
            'hover_duration', 'elapsed_time_diff']

    LEVELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
    level_groups = ["0-4", "5-12", "13-22"]
    DIALOGS = ['that', 'this', 'it', 'you', 'flag', 'can','and','is','the','to']
    name_feature = ['basic', 'undefined', 'close', 'open', 'prev', 'next']
    event_name_feature = ['cutscene_click', 'person_click', 'navigate_click',
        'observation_click', 'notification_click', 'object_click',
        'object_hover', 'map_hover', 'map_click', 'checkpoint',
        'notebook_click'
    ]
    event_name_short = ['notebook_lick', 'map_click', 'object_click', 'observation_click', 'person_click']

    fqid_lists = ['worker', 'archivist', 'gramps', 'wells', 'toentry', 'confrontation', 'crane_ranger', 'groupconvo', 'flag_girl', 'tomap', 'tostacks', 'tobasement', 'archivist_glasses', 'boss', 'journals', 'seescratches', 'groupconvo_flag', 'cs', 'teddy', 'expert', 'businesscards', 'ch3start', 'tunic.historicalsociety', 'tofrontdesk', 'savedteddy', 'plaque', 'glasses', 'tunic.drycleaner', 'reader_flag', 'tunic.library', 'tracks', 'tunic.capitol_2', 'trigger_scarf', 'reader', 'directory', 'tunic.capitol_1', 'journals.pic_0.next', 'unlockdoor', 'tunic', 'what_happened', 'tunic.kohlcenter', 'tunic.humanecology', 'colorbook', 'logbook', 'businesscards.card_0.next', 'journals.hub.topics', 'logbook.page.bingo', 'journals.pic_1.next', 'journals_flag', 'reader.paper0.next', 'tracks.hub.deer', 'reader_flag.paper0.next', 'trigger_coffee', 'wellsbadge', 'journals.pic_2.next', 'tomicrofiche', 'journals_flag.pic_0.bingo', 'plaque.face.date', 'notebook', 'tocloset_dirty', 'businesscards.card_bingo.bingo', 'businesscards.card_1.next', 'tunic.wildlife', 'tunic.hub.slip', 'tocage', 'journals.pic_2.bingo', 'tocollectionflag', 'tocollection', 'chap4_finale_c', 'chap2_finale_c', 'lockeddoor', 'journals_flag.hub.topics', 'tunic.capitol_0', 'reader_flag.paper2.bingo', 'photo', 'tunic.flaghouse', 'reader.paper1.next', 'directory.closeup.archivist', 'intro', 'businesscards.card_bingo.next', 'reader.paper2.bingo', 'retirement_letter', 'remove_cup', 'journals_flag.pic_0.next', 'magnify', 'coffee', 'key', 'togrampa', 'reader_flag.paper1.next', 'janitor', 'tohallway', 'chap1_finale', 'report', 'outtolunch', 'journals_flag.hub.topics_old', 'journals_flag.pic_1.next', 'reader.paper2.next', 'chap1_finale_c', 'reader_flag.paper2.next', 'door_block_talk', 'journals_flag.pic_1.bingo', 'journals_flag.pic_2.next', 'journals_flag.pic_2.bingo', 'block_magnify', 'reader.paper0.prev', 'block', 'reader_flag.paper0.prev', 'block_0', 'door_block_clean', 'reader.paper2.prev', 'reader.paper1.prev', 'doorblock', 'tocloset', 'reader_flag.paper2.prev', 'reader_flag.paper1.prev', 'block_tomap2', 'journals_flag.pic_0_old.next', 'journals_flag.pic_1_old.next', 'block_tocollection', 'block_nelson', 'journals_flag.pic_2_old.next', 'block_tomap1', 'block_badge', 'need_glasses', 'block_badge_2', 'fox', 'block_1']
    text_lists = ['tunic.historicalsociety.cage.confrontation', 'tunic.wildlife.center.crane_ranger.crane', 'tunic.historicalsociety.frontdesk.archivist.newspaper', 'tunic.historicalsociety.entry.groupconvo', 'tunic.wildlife.center.wells.nodeer', 'tunic.historicalsociety.frontdesk.archivist.have_glass', 'tunic.drycleaner.frontdesk.worker.hub', 'tunic.historicalsociety.closet_dirty.gramps.news', 'tunic.humanecology.frontdesk.worker.intro', 'tunic.historicalsociety.frontdesk.archivist_glasses.confrontation', 'tunic.historicalsociety.basement.seescratches', 'tunic.historicalsociety.collection.cs', 'tunic.flaghouse.entry.flag_girl.hello', 'tunic.historicalsociety.collection.gramps.found', 'tunic.historicalsociety.basement.ch3start', 'tunic.historicalsociety.entry.groupconvo_flag', 'tunic.library.frontdesk.worker.hello', 'tunic.library.frontdesk.worker.wells', 'tunic.historicalsociety.collection_flag.gramps.flag', 'tunic.historicalsociety.basement.savedteddy', 'tunic.library.frontdesk.worker.nelson', 'tunic.wildlife.center.expert.removed_cup', 'tunic.library.frontdesk.worker.flag', 'tunic.historicalsociety.frontdesk.archivist.hello', 'tunic.historicalsociety.closet.gramps.intro_0_cs_0', 'tunic.historicalsociety.entry.boss.flag', 'tunic.flaghouse.entry.flag_girl.symbol', 'tunic.historicalsociety.closet_dirty.trigger_scarf', 'tunic.drycleaner.frontdesk.worker.done', 'tunic.historicalsociety.closet_dirty.what_happened', 'tunic.wildlife.center.wells.animals', 'tunic.historicalsociety.closet.teddy.intro_0_cs_0', 'tunic.historicalsociety.cage.glasses.afterteddy', 'tunic.historicalsociety.cage.teddy.trapped', 'tunic.historicalsociety.cage.unlockdoor', 'tunic.historicalsociety.stacks.journals.pic_2.bingo', 'tunic.historicalsociety.entry.wells.flag', 'tunic.humanecology.frontdesk.worker.badger', 'tunic.historicalsociety.stacks.journals_flag.pic_0.bingo', 'tunic.historicalsociety.closet.intro', 'tunic.historicalsociety.closet.retirement_letter.hub', 'tunic.historicalsociety.entry.directory.closeup.archivist', 'tunic.historicalsociety.collection.tunic.slip', 'tunic.kohlcenter.halloffame.plaque.face.date', 'tunic.historicalsociety.closet_dirty.trigger_coffee', 'tunic.drycleaner.frontdesk.logbook.page.bingo', 'tunic.library.microfiche.reader.paper2.bingo', 'tunic.kohlcenter.halloffame.togrampa', 'tunic.capitol_2.hall.boss.haveyougotit', 'tunic.wildlife.center.wells.nodeer_recap', 'tunic.historicalsociety.cage.glasses.beforeteddy', 'tunic.historicalsociety.closet_dirty.gramps.helpclean', 'tunic.wildlife.center.expert.recap', 'tunic.historicalsociety.frontdesk.archivist.have_glass_recap', 'tunic.historicalsociety.stacks.journals_flag.pic_1.bingo', 'tunic.historicalsociety.cage.lockeddoor', 'tunic.historicalsociety.stacks.journals_flag.pic_2.bingo', 'tunic.historicalsociety.collection.gramps.lost', 'tunic.historicalsociety.closet.notebook', 'tunic.historicalsociety.frontdesk.magnify', 'tunic.humanecology.frontdesk.businesscards.card_bingo.bingo', 'tunic.wildlife.center.remove_cup', 'tunic.library.frontdesk.wellsbadge.hub', 'tunic.wildlife.center.tracks.hub.deer', 'tunic.historicalsociety.frontdesk.key', 'tunic.library.microfiche.reader_flag.paper2.bingo', 'tunic.flaghouse.entry.colorbook', 'tunic.wildlife.center.coffee', 'tunic.capitol_1.hall.boss.haveyougotit', 'tunic.historicalsociety.basement.janitor', 'tunic.historicalsociety.collection_flag.gramps.recap', 'tunic.wildlife.center.wells.animals2', 'tunic.flaghouse.entry.flag_girl.symbol_recap', 'tunic.historicalsociety.closet_dirty.photo', 'tunic.historicalsociety.stacks.outtolunch', 'tunic.library.frontdesk.worker.wells_recap', 'tunic.historicalsociety.frontdesk.archivist_glasses.confrontation_recap', 'tunic.capitol_0.hall.boss.talktogramps', 'tunic.historicalsociety.closet.photo', 'tunic.historicalsociety.collection.tunic', 'tunic.historicalsociety.closet.teddy.intro_0_cs_5', 'tunic.historicalsociety.closet_dirty.gramps.archivist', 'tunic.historicalsociety.closet_dirty.door_block_talk', 'tunic.historicalsociety.entry.boss.flag_recap', 'tunic.historicalsociety.frontdesk.archivist.need_glass_0', 'tunic.historicalsociety.entry.wells.talktogramps', 'tunic.historicalsociety.frontdesk.block_magnify', 'tunic.historicalsociety.frontdesk.archivist.foundtheodora', 'tunic.historicalsociety.closet_dirty.gramps.nothing', 'tunic.historicalsociety.closet_dirty.door_block_clean', 'tunic.capitol_1.hall.boss.writeitup', 'tunic.library.frontdesk.worker.nelson_recap', 'tunic.library.frontdesk.worker.hello_short', 'tunic.historicalsociety.stacks.block', 'tunic.historicalsociety.frontdesk.archivist.need_glass_1', 'tunic.historicalsociety.entry.boss.talktogramps', 'tunic.historicalsociety.frontdesk.archivist.newspaper_recap', 'tunic.historicalsociety.entry.wells.flag_recap', 'tunic.drycleaner.frontdesk.worker.done2', 'tunic.library.frontdesk.worker.flag_recap', 'tunic.humanecology.frontdesk.block_0', 'tunic.library.frontdesk.worker.preflag', 'tunic.historicalsociety.basement.gramps.seeyalater', 'tunic.flaghouse.entry.flag_girl.hello_recap', 'tunic.historicalsociety.closet.doorblock', 'tunic.drycleaner.frontdesk.worker.takealook', 'tunic.historicalsociety.basement.gramps.whatdo', 'tunic.library.frontdesk.worker.droppedbadge', 'tunic.historicalsociety.entry.block_tomap2', 'tunic.library.frontdesk.block_nelson', 'tunic.library.microfiche.block_0', 'tunic.historicalsociety.entry.block_tocollection', 'tunic.historicalsociety.entry.block_tomap1', 'tunic.historicalsociety.collection.gramps.look_0', 'tunic.library.frontdesk.block_badge', 'tunic.historicalsociety.cage.need_glasses', 'tunic.library.frontdesk.block_badge_2', 'tunic.kohlcenter.halloffame.block_0', 'tunic.capitol_0.hall.chap1_finale_c', 'tunic.capitol_1.hall.chap2_finale_c', 'tunic.capitol_2.hall.chap4_finale_c', 'tunic.wildlife.center.fox.concern', 'tunic.drycleaner.frontdesk.block_0', 'tunic.historicalsociety.entry.gramps.hub', 'tunic.humanecology.frontdesk.block_1', 'tunic.drycleaner.frontdesk.block_1']
    room_lists = ['tunic.historicalsociety.entry', 'tunic.wildlife.center', 'tunic.historicalsociety.cage', 'tunic.library.frontdesk', 'tunic.historicalsociety.frontdesk', 'tunic.historicalsociety.stacks', 'tunic.historicalsociety.closet_dirty', 'tunic.humanecology.frontdesk', 'tunic.historicalsociety.basement', 'tunic.kohlcenter.halloffame', 'tunic.library.microfiche', 'tunic.drycleaner.frontdesk', 'tunic.historicalsociety.collection', 'tunic.historicalsociety.closet', 'tunic.flaghouse.entry', 'tunic.historicalsociety.collection_flag', 'tunic.capitol_1.hall', 'tunic.capitol_0.hall', 'tunic.capitol_2.hall']


    #　fqid, level, roomと、text, levelで、ある程度レコードが存在する組み合わせをみつける
    if version >= 2:
      if not use_csv:
          session_cnt = x.select('session_id').n_unique()
          low = int(session_cnt * thre) 
          
          flr_list = x.select('fqid', 'level', 'room_fqid', 'session_id').groupby('fqid', 'level', 'room_fqid').n_unique().filter(pl.col('session_id')>=low).drop('session_id')
          single_fqid = flr_list.groupby('fqid').count().rename({'count': 'fqid_count'}).filter(pl.col('fqid_count')==1).get_column('fqid').to_list()
          flr_list = flr_list.filter(~pl.col('fqid').is_in(single_fqid))
          flr_cs = flr_list.get_columns()
          
          tl_list = x.select('text_fqid', 'level', 'session_id').groupby('text_fqid', 'level').n_unique().filter(pl.col('session_id')>=low).drop('session_id')
          single_text = tl_list.groupby('text_fqid').count().rename({'count': 'text_count'}).filter(pl.col('text_count')==1).get_column('text_fqid').to_list()
          tl_list = tl_list.filter(~pl.col('text_fqid').is_in(single_text))
          tl_cs = tl_list.get_columns()
    
      # submissionの場合はこちら
      else:
          flr_list = pl.read_csv(f'{csv_path}/flr_list.csv')
          flr_cs = flr_list.get_columns()

          tl_list = pl.read_csv(f'{csv_path}/tl_list.csv')
          tl_cs = tl_list.get_columns()


    # levelごとの経過時間についての情報をまとめる
    if level_diff:
        df_level_diff = x.groupby('session_id', 'level_group', 'level').agg([
            pl.col('elapsed_time').max().alias('elapsed_time_max'),
            pl.col('elapsed_time').min().alias('elapsed_time_min'),
        ]).sort('session_id', 'level').with_columns([
            (pl.col('elapsed_time_max') - pl.col('elapsed_time_min')).alias('elapsed_time_total'),
            (pl.col('elapsed_time_min').shift(-1)).alias('next_min'),
        ]).with_columns([
            (pl.col('next_min') - pl.col('elapsed_time_max')).alias('level_diff'),
        ])
        df_level_diff_summary = df_level_diff.groupby('session_id').agg([
            *[pl.col('elapsed_time_total').filter(pl.col('level')==l).max().alias(f'elapsed_time_total_level_{l}') for l in LEVELS],
            *[pl.col('level_diff').filter(pl.col('level')==l).max().alias(f'elapsed_time_level_diff_{l}') for l in [i for i in LEVELS if i not in [4,13,22]]],
            (pl.col('elapsed_time_max').filter((pl.col('level')>=0)&(pl.col('level')<=4)).max()).alias('elapsed_time_max0-4'),
            (pl.col('elapsed_time_max').filter((pl.col('level')>=5)&(pl.col('level')<=12)).max()).alias('elapsed_time_max5-12'),
            (pl.col('elapsed_time_max').filter((pl.col('level')>=13)&(pl.col('level')<=22)).max()).alias('elapsed_time_max13-22'),
            (pl.col('elapsed_time_max').filter((pl.col('level')>=0)&(pl.col('level')<=3)).max() - pl.col('elapsed_time_min').filter(pl.col('level')==4).min()).alias('elapsed_time_diff_max0-3_min4'),
            (pl.col('elapsed_time_max').filter((pl.col('level')>=5)&(pl.col('level')<=11)).max() - pl.col('elapsed_time_min').filter(pl.col('level')==12).min()).alias('elapsed_time_diff_max5-11_min12'),
            (pl.col('elapsed_time_max').filter((pl.col('level')>=13)&(pl.col('level')<=21)).max() - pl.col('elapsed_time_min').filter(pl.col('level')==22).min()).alias('elapsed_time_diff_max13-21_min22'),
        ]).with_columns(
            *[pl.col(f'elapsed_time_level_diff_{n}').apply(lambda s: s if s < 0 else 0, return_dtype=pl.Int64).alias(f'elapsed_fime_level_diff_fixed_{n}') for n in [i for i in LEVELS if i not in [4,13,22]]],
            pl.col('elapsed_time_diff_max0-3_min4').apply(lambda s: s if s > 0 else 0, return_dtype=pl.Int64).alias('elapsed_time_diff_fixed_max0_3_min4'),
            pl.col('elapsed_time_diff_max5-11_min12').apply(lambda s: s if s > 0 else 0, return_dtype=pl.Int64).alias('elapsed_time_diff_fixed_max5-11_min12'),
            pl.col('elapsed_time_diff_max13-21_min22').apply(lambda s: s if s > 0 else 0, return_dtype=pl.Int64).alias('elapsed_time_diff_fixed_max13-21_min22'),
        ).drop(
            *[f'elapsed_time_level_diff_{l}' for l in [i for i in range(22) if i not in [4,13,22]]],
            'elapsed_time_diff_max0-3_min4',
            'elapsed_time_diff_max5-11_min12',
            'elapsed_time_diff_max13-21_min22'
        )

    if cut_above:
        # 閾値を作成して結合する
        df_threshold = df_level_diff.groupby('session_id', 'level_group').agg([
            (pl.col('elapsed_time_min').filter(pl.col('level').is_in([4,12,22])).max()).alias('tmp_min')
        ]).with_columns([
            (pl.col('tmp_min') + pl.when(pl.col('level_group')=='0-4').then(pl.lit(860552) 
            ).when(pl.col('level_group')=='5-12').then(pl.lit(1102255)
            ).otherwise(pl.lit(428119))).alias('max_threshold')
        ]).drop('tmp_min')
        x = x.join(df_threshold, on=['session_id', 'level_group'], how='left')

        # 閾値に合わせて分割
        df_train_above = x.filter(pl.col('elapsed_time')>pl.col('max_threshold'))
        x = x.filter(~(pl.col('elapsed_time')>pl.col('max_threshold')))

        # 閾値を超えるカラムの情報
        feature_suffix2 = feature_suffix+'_above'
        df_train_above_summary = df_train_above.groupby('session_id').agg([
            *[pl.col('index').filter(pl.col('room_fqid')==r).count().alias(f'index_count_{r}_{feature_suffix2}') for r in room_lists],
            *[pl.col('index').filter(pl.col('event_name')==e).count().alias(f'index_count_{e}_{feature_suffix2}') for e in event_name_feature],
            *[pl.col('index').filter(pl.col('level')==l).count().alias(f'index_count_{l}_{feature_suffix2}') for l in LEVELS],
            *[pl.col('elapsed_time_diff_to').filter(pl.col('room_fqid')==r).sum().alias(f'elapsed_time_diff_to_{r}_{feature_suffix2}') for r in room_lists],
            *[pl.col('elapsed_time_diff_to').filter(pl.col('event_name')==e).sum().alias(f'elapsed_time_diff_to_{e}_{feature_suffix2}') for e in event_name_feature],
            *[pl.col('elapsed_time_diff_to').filter(pl.col('level')==l).sum().alias(f'elapsed_time_diff_to_{l}_{feature_suffix2}') for l in LEVELS]
        ])

    # メインの処理
    aggs = [
        pl.col("index").count().alias(f"session_number_{feature_suffix}"),
      
        *[pl.col(c).drop_nulls().n_unique().alias(f"{c}_unique_{feature_suffix}") for c in CATS],
        *[pl.col(c).quantile(0.1, "nearest").alias(f"{c}_quantile1_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.2, "nearest").alias(f"{c}_quantile2_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.4, "nearest").alias(f"{c}_quantile4_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.6, "nearest").alias(f"{c}_quantile6_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.8, "nearest").alias(f"{c}_quantile8_{feature_suffix}") for c in NUMS],
        *[pl.col(c).quantile(0.9, "nearest").alias(f"{c}_quantile9_{feature_suffix}") for c in NUMS],
        
        *[pl.col(c).mean().alias(f"{c}_mean_{feature_suffix}") for c in NUMS],
        *[pl.col(c).std().alias(f"{c}_std_{feature_suffix}") for c in NUMS],
        *[pl.col(c).min().alias(f"{c}_min_{feature_suffix}") for c in NUMS],
        *[pl.col(c).max().alias(f"{c}_max_{feature_suffix}") for c in NUMS],
        
        *[pl.col("event_name").filter(pl.col("event_name") == c).count().alias(f"{c}_event_name_counts{feature_suffix}")for c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).quantile(0.1, "nearest").alias(f"{c}_ET_quantile1_{feature_suffix}") for c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).quantile(0.2, "nearest").alias(f"{c}_ET_quantile2_{feature_suffix}") for c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).quantile(0.4, "nearest").alias(f"{c}_ET_quantile4_{feature_suffix}") for c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).quantile(0.6, "nearest").alias(f"{c}_ET_quantile6_{feature_suffix}") for c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).quantile(0.8, "nearest").alias(f"{c}_ET_quantile8_{feature_suffix}") for c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).quantile(0.9, "nearest").alias(f"{c}_ET_quantile9_{feature_suffix}") for c in event_name_feature],      
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in event_name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("event_name")==c).min().alias(f"{c}_ET_min_{feature_suffix}") for c in event_name_feature],
     
        *[pl.col("name").filter(pl.col("name") == c).count().alias(f"{c}_name_counts{feature_suffix}")for c in name_feature],   
        *[pl.col("elapsed_time_diff").filter(pl.col("name")==c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("name")==c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("name")==c).min().alias(f"{c}_ET_min_{feature_suffix}") for c in name_feature],
        *[pl.col("elapsed_time_diff").filter(pl.col("name")==c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in name_feature],  
        
        *[pl.col("room_fqid").filter(pl.col("room_fqid") == c).count().alias(f"{c}_room_fqid_counts{feature_suffix}")for c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).min().alias(f"{c}_ET_min_{feature_suffix}") for c in room_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("room_fqid") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in room_lists],
                
        *[pl.col("fqid").filter(pl.col("fqid") == c).count().alias(f"{c}_fqid_counts{feature_suffix}")for c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).min().alias(f"{c}_ET_min_{feature_suffix}") for c in fqid_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("fqid") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in fqid_lists],
       
        *[pl.col("text_fqid").filter(pl.col("text_fqid") == c).count().alias(f"{c}_text_fqid_counts{feature_suffix}") for c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).max().alias(f"{c}_ET_max_{feature_suffix}") for c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).min().alias(f"{c}_ET_min_{feature_suffix}") for c in text_lists],
        *[pl.col("elapsed_time_diff").filter(pl.col("text_fqid") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in text_lists],
         
        *[pl.col("location_x_diff").filter(pl.col("event_name")==c).mean().alias(f"{c}_ET_mean_x{feature_suffix}") for c in event_name_feature],
        *[pl.col("location_x_diff").filter(pl.col("event_name")==c).std().alias(f"{c}_ET_std_x{feature_suffix}") for c in event_name_feature],
        *[pl.col("location_x_diff").filter(pl.col("event_name")==c).max().alias(f"{c}_ET_max_x{feature_suffix}") for c in event_name_feature],
        *[pl.col("location_x_diff").filter(pl.col("event_name")==c).min().alias(f"{c}_ET_min_x{feature_suffix}") for c in event_name_feature],

        *[pl.col("level").filter(pl.col("level") == c).count().alias(f"{c}_LEVEL_count{feature_suffix}") for c in LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in LEVELS],
        *[pl.col("elapsed_time_diff").filter(pl.col("level") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in LEVELS],

        *[pl.col('index').filter(pl.col('text').str.contains(c)).count().alias(f'word_{c}_{feature_suffix}') for c in DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).mean().alias(f'word_mean_{c}_{feature_suffix}') for c in DIALOGS],
        *[pl.col("elapsed_time_diff").filter((pl.col('text').str.contains(c))).std().alias(f'word_std_{c}_{feature_suffix}') for c in DIALOGS],
        *[pl.col("elapsed_time_diff").filter(pl.col('text').str.contains(c)).max().alias(f'word_max_{c}_{feature_suffix}') for c in DIALOGS],
        *[pl.col("elapsed_time_diff").filter(pl.col('text').str.contains(c)).sum().alias(f'word_sum_{c}_{feature_suffix}') for c in DIALOGS],

        *[pl.col("level_group").filter(pl.col("level_group") == c).count().alias(f"{c}_LEVEL_group_count{feature_suffix}") for c in level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).std().alias(f"{c}_ET_std_{feature_suffix}") for c in level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).mean().alias(f"{c}_ET_mean_{feature_suffix}") for c in level_groups],
        *[pl.col("elapsed_time_diff").filter(pl.col("level_group") == c).sum().alias(f"{c}_ET_sum_{feature_suffix}") for c in level_groups],

    ]
    
    df = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")
  
    if use_extra:
        if grp=='0-4':
            aggs = [
                pl.col("elapsed_time").filter((pl.col("text")=="It's a women's basketball jersey!")|(pl.col("text")=="That settles it.")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("shirt_watch_duration"),
                pl.col("index").filter((pl.col("text")=="It's a women's basketball jersey!")|(pl.col("text")=="That settles it.")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("shirt_watch_indexCount"),
                pl.col("elapsed_time").filter((pl.col("text")=="Now where did I put my notebook?")|(pl.col("text")=="Found it!")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("notebook_found_duration"),
                pl.col("index").filter((pl.col("text")=="Now where did I put my notebook?")|(pl.col("text")=="Found it!")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("notebook_found_indexCount"),

                pl.col("elapsed_time").filter(pl.col("text")=="Hey Jo, let's take a look at the shirt!").max().alias("to_shirt_be_et"),
                pl.col("index").filter(pl.col("text")=="Hey Jo, let's take a look at the shirt!").max().alias("to_shirt_be_id"),
                pl.col("elapsed_time").filter(pl.col("text_fqid")=="tunic.historicalsociety.collection.cs").min().alias("to_shirt_af_et"),
                pl.col("index").filter(pl.col("text_fqid")=="tunic.historicalsociety.collection.cs").min().alias("to_shirt_af_id"),
            ]
            tmp = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")

            columns = [
              (pl.col('to_shirt_af_et') - pl.col('to_shirt_be_et')).alias('to_shirt_et'),
              (pl.col('to_shirt_af_id') - pl.col('to_shirt_be_id')).alias('to_shirt_id')
            ]
            tmp = tmp.with_columns(columns).drop('to_shirt_af_et', 'to_shirt_af_id', 'to_shirt_be_et', 'to_shirt_be_id')

            df = df.join(tmp, on="session_id", how='left')

        if grp=='5-12':
            aggs = [
                pl.col("elapsed_time").filter((pl.col("text")=="Here's the log book.")|(pl.col("fqid")=='logbook.page.bingo')).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("logbook_bingo_duration"),
                pl.col("index").filter((pl.col("text")=="Here's the log book.")|(pl.col("fqid")=='logbook.page.bingo')).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("logbook_bingo_indexCount"),
                pl.col("elapsed_time").filter((pl.col("fqid")=="businesscards")|(pl.col("fqid")=='businesscards.card_bingo.bingo')).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("businesscard_bingo_duration"),
                pl.col("index").filter((pl.col("fqid")=="businesscards")|(pl.col("fqid")=='businesscards.card_bingo.bingo')).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("businesscard_bingo_indexCount"),
                pl.col("elapsed_time").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='reader'))|(pl.col("fqid")=="reader.paper2.bingo")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("reader_bingo_duration"),
                pl.col("index").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='reader'))|(pl.col("fqid")=="reader.paper2.bingo")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("reader_bingo_indexCount"),
                pl.col("elapsed_time").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='journals'))|(pl.col("fqid")=="journals.pic_2.bingo")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("journals_bingo_duration"),
                pl.col("index").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='journals'))|(pl.col("fqid")=="journals.pic_2.bingo")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("journals_bingo_indexCount"),
                pl.col("elapsed_time").filter((pl.col("text")=="Hmmm... not sure. Why don't you try the library?")|(pl.col("text")=="Oh, hello there!")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("go_to_library_duration"), # Level 9 図書館への移動
                pl.col("index").filter((pl.col("text")=="Hmmm... not sure. Why don't you try the library?")|(pl.col("text")=="Oh, hello there!")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("go_to_library_indexCount"),
                pl.col("elapsed_time").filter((pl.col("text")=="You could ask the archivist. He knows everybody!")|(pl.col("text")=="Do you have any info on Theodora Youmans?")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("go_to_archivist_duration"), # Level 11 文書館への移動
                pl.col("index").filter((pl.col("text")=="You could ask the archivist. He knows everybody!")|(pl.col("text")=="Do you have any info on Theodora Youmans?")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("go_to_archivist_indexCount"),

                # Level 6: メガネ
                pl.col("elapsed_time").filter(pl.col("text_fqid")=="tunic.historicalsociety.frontdesk.archivist.hello").max().alias("found_glasses_be_et"), 
                pl.col("index").filter(pl.col("text_fqid")=="tunic.historicalsociety.frontdesk.archivist.hello").max().alias("found_glasses_be_id"),
                pl.col("elapsed_time").filter(pl.col("text_fqid")=="tunic.historicalsociety.frontdesk.magnify").min().alias("found_glasses_af_et"),
                pl.col("index").filter(pl.col("text_fqid")=="tunic.historicalsociety.frontdesk.magnify").min().alias("found_glasses_af_id"),                
            ]
            tmp = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")

            columns = [
              (pl.col('found_glasses_af_et') - pl.col('found_glasses_be_et')).alias('found_glasses_et'),
              (pl.col('found_glasses_af_id') - pl.col('found_glasses_be_id')).alias('found_glasses_id')
            ]
            tmp = tmp.with_columns(columns).drop('found_glasses_af_et', 'found_glasses_af_id', 'found_glasses_be_et', 'found_glasses_be_id')

            df = df.join(tmp, on="session_id", how='left')

        if grp=='13-22':
            aggs = [
                pl.col("elapsed_time").filter((pl.col("fqid")=="I'll go look at everyone's pictures!")|(pl.col("text")=="Those are the same glasses!")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l15_glasses_bingo_duration"),
                pl.col("index").filter((pl.col("text")=="I'll go look at everyone's pictures!")|(pl.col("text")=="Those are the same glasses!")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l15_glasses_bingo_indexCount"),              
                pl.col("elapsed_time").filter((pl.col("text")=="Yes! It's the key for Teddy's cage!")|(pl.col("text")=="Those are the same glasses!")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l15_archivist_key_duration"),
                pl.col("index").filter((pl.col("text")=="Yes! It's the key for Teddy's cage!")|(pl.col("text")=="Those are the same glasses!")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l15_archivist_key_indexCount"),
                pl.col("elapsed_time").filter((pl.col("text")=="Your grampa is waiting for you in the collection room.")|(pl.col("text")=="The archivist had him locked up!")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l17_grampa_wating_duration"),
                pl.col("index").filter((pl.col("text")=="Your grampa is waiting for you in the collection room.")|(pl.col("text")=="The archivist had him locked up!")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l17_grampa_wating_indexCount"),
                pl.col("elapsed_time").filter((pl.col("text")=="Go take a look!")|(pl.col("text")=="That hoofprint doesn't match the flag!")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l18_hoofprint_bingo_duration"),
                pl.col("index").filter((pl.col("text")=="Go take a look!")|(pl.col("text")=="That hoofprint doesn't match the flag!")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l18_hoofprint_indexCount"),
                pl.col("elapsed_time").filter((pl.col("text")=="Hmm. You could try the Aldo Leopold Wildlife Center.")|(pl.col("text")=="Oh no! What happened to that crane?")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l18_go_to_wildlife_duration"),
                pl.col("index").filter((pl.col("text")=="Hmm. You could try the Aldo Leopold Wildlife Center.")|(pl.col("text")=="Oh no! What happened to that crane?")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l18_go_to_wildlife_indexCount"),
                pl.col("elapsed_time").filter((pl.col("text")=="Hey, nice dog! What breed is he?")|(pl.col("text")=="Actually, I went to school with somebody who LOVES old flags.")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l19_go_to_flag_expert_duration"),
                pl.col("index").filter((pl.col("text")=="Hey, nice dog! What breed is he?")|(pl.col("text")=="Actually, I went to school with somebody who LOVES old flags.")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l19_go_to_flag_expert_indexCount"),
                pl.col("elapsed_time").filter((pl.col("text")=="It's an ecology flag!")|(pl.col("text")=="Hey, I've seen that symbol before! Check it out!")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l19_ecologyflag_check_duration"),
                pl.col("index").filter((pl.col("text")=="It's an ecology flag!")|(pl.col("text")=="Hey, I've seen that symbol before! Check it out!")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l19_ecologyflag_check_indexCount"),
                pl.col("elapsed_time").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='reader_flag'))|(pl.col("fqid")=="tunic.library.microfiche.reader_flag.paper2.bingo")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l20_reader_flag_duration"),
                pl.col("index").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='reader_flag'))|(pl.col("fqid")=="tunic.library.microfiche.reader_flag.paper2.bingo")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l20_reader_flag_indexCount"),
                pl.col("elapsed_time").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='journals_flag'))|(pl.col("fqid")=="journals_flag.pic_0.bingo")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l21_journalsFlag_bingo_duration"),
                pl.col("index").filter(((pl.col("event_name")=='navigate_click')&(pl.col("fqid")=='journals_flag'))|(pl.col("fqid")=="journals_flag.pic_0.bingo")).apply(lambda s: s.max()-s.min() if s.len()>0 else 0).alias("l21_journalsFlag_bingo_indexCount"),
            ]
            tmp = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")
            df = df.join(tmp, on="session_id", how='left')
    
    # 特徴量を更に追加する
    if version>=2:

        # 集計統計量の追加
        aggs = [
            
            # levelと主要イベントの組での回数
            *[pl.col("event_name").filter((pl.col("event_name")==c)&(pl.col("level")==l)).count().alias(f"{c}_{l}_ET_min_{feature_suffix}") for c in event_name_short for l in LEVELS],
            
            # fqid, room, levelの組での経過時間と回数
            *[pl.col("elapsed_time_diff").filter((pl.col("fqid") == f)&(pl.col("room_fqid")==r)&(pl.col("level")==l)).sum().alias(f"{f}_{r}_{l}_ET_sum_{feature_suffix}") for f, l, r in zip(flr_cs[0], flr_cs[1], flr_cs[2])],
            *[pl.col("index").filter((pl.col("fqid") == f)&(pl.col("room_fqid")==r)&(pl.col("level")==l)).count().alias(f"{f}_{r}_{l}_counts{feature_suffix}") for f, l, r in zip(flr_cs[0], flr_cs[1], flr_cs[2])],
            
            # text, levelの組での経過時間と回数
            *[pl.col("elapsed_time_diff").filter((pl.col("text_fqid") == t)&(pl.col('level')==l)).sum().alias(f"{t}_{l}_ET_sum_{feature_suffix}") for t, l in zip(tl_cs[0], tl_cs[1])],
            *[pl.col("index").filter((pl.col("text_fqid") == t)&(pl.col('level')==l)).count().alias(f"{t}_{l}ET_count_{feature_suffix}") for t, l in zip(tl_cs[0], tl_cs[1])],

            # nameとevent_nameの組み合わせ
            *[pl.col("index").filter((pl.col("event_name")==e)&(pl.col("name")==n)).count().alias(f"{e}_{n}_count_{feature_suffix}") for e in ["map_click", "notebook_click"] for n in name_feature]
        ]
        tmp = x.groupby(["session_id"], maintain_order=True).agg(aggs).sort("session_id")
        df = df.join(tmp, on="session_id", how='left')
        
        # 不審なクリック
        click_list = x.filter(pl.col('screen_coor_x').is_not_null()).groupby('screen_coor_x', 'screen_coor_y', 'session_id').count().sort('count', descending=True)
        tmp = click_list.groupby('session_id').max().select('session_id', pl.col('count').alias(f'click_same_max_{feature_suffix}'))
        tmp2 = click_list.filter(pl.col('count')>=5).groupby('session_id').count().select('session_id', pl.col('count').alias(f'click_same_over5_count_{feature_suffix}'))
        tmp = tmp.join(tmp2, on='session_id', how='left')
        df = df.join(tmp, on='session_id', how='left')
    
    if level_diff:
        df = df.join(df_level_diff_summary, on='session_id', how='left')
    if cut_above:
        df = df.join(df_train_above_summary, on='session_id', how='left')
        
    return df


def add_random_feature(df, n=10):
    
    height = df.shape[0]
    data = np.random.randint(1, 1000, size=(height, n))

    df_rand = pl.DataFrame(data=data, schema=[f'random{i}' for i in range(n)])
    
    df = pl.concat([df, df_rand], how='horizontal')
    
    return df


def add_columns_session(df, id=6):

    time_columns = [
        pl.col('session_id').apply(lambda x: int(str(x)[:2])).alias('year'),
        pl.col('session_id').apply(lambda x: int(str(x)[2:4])+1).alias('month'),
        pl.col('session_id').apply(lambda x: int(str(x)[4:6])).alias('day'),
        pl.col('session_id').apply(lambda x: int(str(x)[6:8])).alias('hour'),
        pl.col('session_id').apply(lambda x: int(str(x)[8:10])).alias('minute'),
        pl.col('session_id').apply(lambda x: int(str(x)[10:12])).alias('second')
    ][:id]

    df = df.with_columns(*time_columns)
  
    return df