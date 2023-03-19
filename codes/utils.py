import pandas as pd
import numpy as np
import polars as pl

def pl_to_pd(df, index_col='session_id'):
    return df.to_pandas().set_index(index_col)