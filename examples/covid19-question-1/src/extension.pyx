import numpy as np
import pandas as pd

cdef f_gen_row(idx):
    def f(x):
        return idx[x]
    return f

cdef f_gen_col(col, idx):
    def f(x):
        return idx[(x, col)]
    return f

def scatter_features(df, group_by, feature_col, dictionary={}, aggregate_funcs=None, selected_feats=[]):
    key_cols = (group_by + [feature_col]) if type(group_by) is list else [group_by, feature_col]
    other_cols = [col for col in df.columns if col not in key_cols]
    if len(other_cols) == 0:
        other_cols = ['occurrence']
        df['occurrence'] = [1] * df.shape[0]
    if len(selected_feats) > 0: df = df.loc[df[feature_col].isin(selected_feats)]
    rows = df[group_by].drop_duplicates().values
    cols = [(prefix, col) for prefix in df[feature_col].drop_duplicates().values for col in other_cols]
    row_idx, col_idx = dict(zip(rows, range(len(rows)))), dict(zip(cols, range(len(cols))))
    ridx = df[group_by].apply(f_gen_row(row_idx)).values
    data = np.ones((len(rows),len(cols))) * np.nan
    cdef double[:, :] data_view = data
    for col in other_cols:
        cidx = df[feature_col].apply(f_gen_col(col, col_idx)).values
        data_view.base[ridx, cidx] = df[col].astype('double').values
    return pd.DataFrame(data, index=rows, columns=['_'.join([dictionary.setdefault(str(prefix), '_'.join([feature_col, str(prefix)])).replace(' ', '_'), col]) for prefix, col in cols])
