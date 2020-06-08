import os, time, itertools, multiprocessing
import datetime as dt

import numpy as np
import pandas as pd
import dask.dataframe as dd
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture

# from extension import scatter_features as c_scatter_features

DATA_PATH = '/data'
NOW = dt.datetime.now()


def split_1d(task_num, split_num=None, task_size=None, split_size=None, ret_idx=False):
    if (split_num is None):
        if (task_size is None or split_size is None):
            return list(range(task_num+1)) if ret_idx else [1] * task_num
        group_size = max(1, int(split_size / task_size))
        split_num = int(task_num / group_size)
        remainder = task_num % group_size
        results = [group_size] * split_num if (remainder == 0) else [group_size] * split_num + [remainder]
    else:
        group_size = int(task_num / split_num)
        remainder = task_num % split_num
        results = [group_size] * split_num if (remainder == 0) else [group_size + 1] * remainder + [group_size] * (split_num - remainder)
    return np.cumsum([0]+results).tolist() if ret_idx else results

def run_pool(target, n_jobs=1, pool=None, ret_pool=False, dist_param=[], **kwargs):
    if (n_jobs < 1): return None
    res_list, fix_kwargs, iter_kwargs = [], {}, {}
    if (pool is None):
        pool = multiprocessing.Pool(processes=n_jobs)
    # Separate iterable arguments and singular arguments
    for k, v in kwargs.items():
        if (k in dist_param and hasattr(v, '__iter__')):
            iter_kwargs[k] = v
        else:
            fix_kwargs[k] = v
    if (len(iter_kwargs) == 0):
        res_list = [pool.apply_async(target, kwds=kwargs) for x in range(n_jobs)]
    else:
        # Construct arguments for each process
        for argv in zip(*iter_kwargs.values()):
            args = dict(zip(iter_kwargs.keys(), argv))
            args.update(fix_kwargs)
            r = pool.apply_async(target, kwds=args)
            res_list.append(r)
    res_list = [r.get() for r in res_list] if len(res_list) > 1 else res_list[0].get()
    time.sleep(0.01)
    if (ret_pool):
        return res_list, pool
    pool.close()
    pool.join()
    return res_list

def fix_duplicates(X, suffix=[]):
    unique_X = {}
    for i, x in enumerate(X): unique_X.setdefault(x, []).append(i)
    return [x if len(unique_X[x])==1 else x+'_%s'%unique_X[x].index(i) for i, x in enumerate(X)] if len(suffix) != len(X) else [x if len(unique_X[x])==1 else x+'_%s'%suffix[i] for i, x in enumerate(X)]

def filter_cols(df, force=[], keywords=[]):
    return [col for col in df.columns if col not in force and not any([k in col for k in keywords]) and df[col].value_counts().shape[0] > 1]

def normalize(a, ord=1):
    norm=np.linalg.norm(a, ord=ord)
    if norm==0: norm=np.finfo(a.dtype).eps
    return a/norm

get_delta_days = lambda x: x.days + x.seconds / 86400

def era_features(start_dates, end_dates, focus_date=dt.datetime.now(), feature_name=''):
    start2focus = start_dates.apply(lambda x: get_delta_days(focus_date - x)).rename((feature_name+'_start2focus_timedelta') if feature_name else 'start2focus_timedelta')
    end2focus = end_dates.apply(lambda x: get_delta_days(focus_date - x)).rename((feature_name+'_end2focus_timedelta') if feature_name else 'end2focus_timedelta')
    start2end = (end_dates - start_dates).apply(get_delta_days).rename((feature_name+'_start2end_timedelta') if feature_name else 'start2end_timedelta')
    return pd.concat([start2focus, end2focus, start2end], axis=1)

def scatter_features_orig(df, group_by, feature_col, dictionary={}, aggregate_funcs=None):
    key_cols = (group_by + [feature_col]) if type(group_by) is list else [group_by, feature_col]
    other_cols = [col for col in df.columns if col not in key_cols]
    data = []
    for gid, grp in df.groupby(group_by):
        grp = grp.groupby(feature_col).agg(np.mean if aggregate_funcs is None else aggregate_funcs).reset_index()
        fixed_other_cols = [col for col in other_cols if col in grp.columns]
        feature_prefixs = [dictionary.setdefault(str(x), '_'.join([feature_col, str(x)])).replace(' ', '_') for x in grp[feature_col]]
        feature_prefixs = fix_duplicates(feature_prefixs, suffix=list(map(str, grp[feature_col])))
        feature_names = ['_'.join([prefix, col]) for prefix in feature_prefixs for col in fixed_other_cols]
        data.append(pd.DataFrame(grp[fixed_other_cols].values.reshape((1,-1)), index=[gid], columns=feature_names))
    return pd.concat(data, axis=0)

def scatter_features(df, group_by, feature_col, dictionary={}, aggregate_funcs=None, selected_feats=[]):
    if len(selected_feats) > 0: df = df.loc[df[feature_col].isin(selected_feats)]
    if type(df) is not pd.DataFrame and type(df) is dd.DataFrame: df = df.compute()
    key_cols = (group_by + [feature_col]) if type(group_by) is list else [group_by, feature_col]
    other_cols = [col for col in df.columns if col not in key_cols]
    if len(other_cols) == 0:
        other_cols = ['occurrence']
        df['occurrence'] = [1] * len(df)
    rows = df[group_by].drop_duplicates().values
    cols = [(prefix, col) for prefix in df[feature_col].drop_duplicates().values for col in other_cols]
    row_idx, col_idx = dict(zip(rows, range(len(rows)))), dict(zip(cols, range(len(cols))))
    data = np.ones((len(rows),len(cols))) * np.nan
    ridx = [row_idx[i] for i in df[group_by].values]
    for col in other_cols:
        data[ridx, [col_idx[j] for j in df[feature_col].apply(lambda x: (x, col)).values]] = df[col]
    return pd.DataFrame(data, index=rows, columns=['_'.join([dictionary.setdefault(str(prefix), '_'.join([feature_col, str(prefix)])).replace(' ', '_'), col]) for prefix, col in cols])

def _func_gen_flatcol(col, idx):
    def f(x):
        return idx[(x, col)] if x != 'foo' else -1
    return f

def _func_gen_flatcol_part(fcol, col, idx):
    def f(df):
        return df[fcol].apply(_func_gen_flatcol(col, idx))
    return f

def scatter_features_dask(df, group_by, feature_col, dictionary={}, aggregate_funcs=None, selected_feats=[]):
    if len(selected_feats) > 0: df = df.loc[df[feature_col].isin(selected_feats)]
    key_cols = (group_by + [feature_col]) if type(group_by) is list else [group_by, feature_col]
    other_cols = [col for col in df.columns if col not in key_cols]
    if len(other_cols) == 0:
        other_cols = ['occurrence']
        df = df.reset_index().set_index('index')
        df = df.map_partitions(lambda subdf: subdf.assign(occurrence=np.ones(len(subdf), dtype='int8')))
    df[feature_col] = df[feature_col].fillna(df[feature_col].value_counts().index.compute()[0])
    rows = df[group_by].drop_duplicates().dropna().compute()
    df = df.loc[df[group_by].isin(rows)]
    prefixs = df[feature_col].drop_duplicates().compute()
    cols = [(prefix, col) for prefix in prefixs for col in other_cols]
    row_idx, col_idx = dict(zip(rows, range(len(rows)))), dict(zip(cols, range(len(cols))))
    ridx = df.map_partitions(lambda subdf: subdf[group_by].apply(lambda x: row_idx[x])).values.compute()
    data = np.ones((len(rows),len(cols))) * np.nan
    for col in other_cols:
        cidx = df.map_partitions(_func_gen_flatcol_part(feature_col, col, col_idx)).values
        t0 = time.time()
        data[ridx, cidx] = df[col].compute()
        t0 = time.time()
    return pd.DataFrame(data, index=rows, columns=['_'.join([dictionary.setdefault(str(prefix), '_'.join([feature_col, str(prefix)])).replace(' ', '_'), col]) for prefix, col in cols])

def _agg_feat(df, group_by, feature_col, other_cols, dictionary={}, aggregate_funcs=None, pid=0):
    data = []
    for gid, grp in df.groupby(group_by):
        grp = grp.groupby(feature_col).agg(np.mean if aggregate_funcs is None else aggregate_funcs).reset_index()
        fixed_other_cols = [col for col in other_cols if col in grp.columns]
        feature_prefixs = [dictionary.setdefault(str(x), '_'.join([feature_col, str(x)])).replace(' ', '_') for x in grp[feature_col]]
        feature_prefixs = fix_duplicates(feature_prefixs, suffix=list(map(str, grp[feature_col])))
        feature_names = ['_'.join([prefix, col]) for prefix in feature_prefixs for col in fixed_other_cols]
        data.append(pd.DataFrame(grp[fixed_other_cols].values.reshape((1,-1)), index=[gid], columns=feature_names))
    return data

def scatter_features_par(df, group_by, feature_col, dictionary={}, aggregate_funcs=None, n_jobs=1):
    key_cols = (group_by + [feature_col]) if type(group_by) is list else [group_by, feature_col]
    other_cols = [col for col in df.columns if col not in key_cols]
    if n_jobs==1:
        data = _agg_feat(df, group_by, feature_col, other_cols, dictionary=dictionary, aggregate_funcs=aggregate_funcs)
    else:
        indices = np.unique(df[group_by].values)
        task_bnd = split_1d(len(indices), split_num=n_jobs, ret_idx=True)
        df = df.set_index(group_by)
        all_data = run_pool(_agg_feat, n_jobs=n_jobs, dist_param=['df', 'pid'], df=[df.loc[indices[task_bnd[i]:task_bnd[i+1]]].reset_index() for i in range(n_jobs)], pid=range(n_jobs), group_by=group_by, feature_col=feature_col, other_cols=other_cols, dictionary=dictionary, aggregate_funcs=aggregate_funcs)
        df = df.reset_index()
        data = itertools.chain.from_iterable(all_data)
    return pd.concat(data, axis=0).loc[df[group_by].drop_duplicates().values]

def encode_cat_features(df, cat_cols, inplace=False):
    encoding = {}
    if type(cat_cols) is list:
        for col in cat_cols:
            df[col].fillna(df[col][df[col].notnull()].value_counts().values.argmax() if df[col].dtypes==object else df[col][df[col].notnull()].median(), inplace=True)
            encoding[col] = dict([(v, i) for i, v in enumerate(np.unique(df[col]))])
    elif type(cat_cols) is str:
        df[cat_cols].fillna(df[cat_cols][df[cat_cols].notnull()].value_counts().values.argmax() if df[cat_cols].dtypes==object else df[cat_cols][df[cat_cols].notnull()].median(), inplace=True)
        encoding = dict([(v, i) for i, v in enumerate(np.unique(df[cat_cols]))])
    return df.replace(encoding, inplace=inplace)


print('Loading measurement.csv ...', flush=True)
t0 = time.time()
measurement = dd.read_csv(os.path.join(DATA_PATH, 'measurement.csv'), usecols=['measurement_concept_id','value_as_number','person_id'])
measurement = measurement.dropna(subset = ['measurement_concept_id'])
measurement = measurement.astype({"measurement_concept_id": str})
print('Data load time: %0.3fs' % (time.time() - t0))
print('Loading condition.csv ...', flush=True)
t0 = time.time()
condition = dd.read_csv(os.path.join(DATA_PATH, "condition_occurrence.csv"), usecols=['condition_concept_id','person_id'])
condition = condition.dropna(subset = ['condition_concept_id'])
condition = condition.astype({"condition_concept_id": str})
print('Data load time: %0.3fs' % (time.time() - t0))
# print('Loading condition_era.csv ...', flush=True)
# t0 = time.time()
# condition_era = dd.read_csv(os.path.join(DATA_PATH, 'condition_era.csv'), dtype={'condition_concept_id':object, 'condition_era_id':int}, parse_dates=['condition_era_start_date', 'condition_era_end_date'], infer_datetime_format=True)
# print('Data load time: %0.3fs' % (time.time() - t0))
print('Loading person.csv ...', flush=True)
t0 = time.time()
person = pd.read_csv(os.path.join(DATA_PATH, 'person.csv'), parse_dates=['birth_datetime'], infer_datetime_format=True)
print('Data load time: %0.3fs' % (time.time() - t0))
print('Loading location.csv ...', flush=True)
t0 = time.time()
location = pd.read_csv(os.path.join(DATA_PATH, 'location.csv'))
print('Data load time: %0.3fs' % (time.time() - t0))
print('Loading data_dictionary.csv ...', flush=True)
t0 = time.time()
data_dictionary = dd.read_csv(os.path.join(DATA_PATH, 'data_dictionary.csv')).set_index('concept_id')
dictionary = dict([(str(k), v) for k, v in data_dictionary.concept_name.iteritems()])
print('Data load time: %0.3fs' % (time.time() - t0))

pre_selected_feats = dict(measurement=list(map(str, [3020891, 3027018, 3012888, 3004249, 3023314, 3013650, 3004327, 3016502])), condition=list(map(str, [254761, 259153, 378253, 437663])))

print('Generating features...', flush=True)
t0 = time.time()
measurement_features = scatter_features_dask(measurement, 'person_id', 'measurement_concept_id', dictionary=dictionary, selected_feats=pre_selected_feats.setdefault('measurement', []))
measurement_features = measurement_features[filter_cols(measurement_features)]
print('Measurement features cost time: %0.3fs' % (time.time() - t0))
t0 = time.time()
# condition_all = pd.merge(condition, condition_era, how='outer', on=['person_id', 'condition_concept_id']).drop_duplicates()
# condition_era_features = era_features(condition_all.condition_era_start_date, condition_all.condition_era_end_date, focus_date=NOW, feature_name='condition')
# condition_df = pd.concat([condition_all[filter_cols(condition_all, force=['condition_era_id'], keywords=['date'])], condition_era_features], axis=1)
condition_features = scatter_features_dask(condition, 'person_id', 'condition_concept_id', dictionary=dictionary, selected_feats=pre_selected_feats.setdefault('condition', []))
condition_features = condition_features[filter_cols(condition_features)]
print('Condition features cost time: %0.3fs' % (time.time() - t0))
t0 = time.time()
person_birthdate = person.birth_datetime.fillna(dt.datetime.fromtimestamp(person.birth_datetime[person.birth_datetime.notnull()].apply(dt.datetime.timestamp).median())).apply(lambda x: get_delta_days(NOW - x))
person_df = pd.concat([person[filter_cols(person, force=['year_of_birth', 'month_of_birth', 'day_of_birth'], keywords=['date'])], person_birthdate.to_frame()], axis=1)
person_df = pd.merge(person_df, location, how='left', on='location_id')
person_features = person_df[filter_cols(person_df)].set_index('person_id')
encode_cat_features(person_features, ['gender_concept_id', 'race_concept_id', 'ethnicity_concept_id', 'gender_source_value', 'race_source_concept_id', 'ethnicity_source_concept_id'], inplace=True)
encode_cat_features(person_features, 'location_id', inplace=True)
print('Person features cost time: %0.3fs' % (time.time() - t0))

print('Preparing data...', flush=True)
t0 = time.time()
X = pd.concat([person_features, measurement_features, condition_features], axis=1, join='outer').replace([np.inf, -np.inf], np.nan)
print('Cost time: %0.3fs' % (time.time() - t0))

print('Filling empty values...', flush=True)
t0 = time.time()
processed_cols = []
condition_value_cols = [col for col in X.columns if col.endswith('condition_occurrence_count')]
processed_cols.extend(condition_value_cols)
for col in condition_value_cols: X[col].fillna(0, inplace=True)
for col in set(X.columns)-set(processed_cols): X[col].fillna(X[col][X[col].notnull()].median(), inplace=True)
print('Cost time: %0.3fs' % (time.time() - t0))

print('Fitting into the model...', flush=True)
t0 = time.time()
# K-Means
clt_model = MiniBatchKMeans(n_clusters=2, random_state=0)
cdists = clt_model.fit_transform(X)
y_proba = np.vstack([1 - normalize(d) for d in cdists])[:,1]
# Gaussian Mixture
# clt_model = GaussianMixture(n_components=2, random_state=0)
# clt_model.fit(X)
# y_proba = clt_model.predict_proba(X)[:,1]
print('Cost time: %0.3fs' % (time.time() - t0))

print('Generating the predictions...', flush=True)
t0 = time.time()
predictions = pd.concat([person.person_id, pd.DataFrame(y_proba, columns=['score'])], axis=1, ignore_index=False)
predictions.to_csv('/output/predictions.csv', index=False)
print('Cost time: %0.3fs' % (time.time() - t0))
print('Done!', flush=True)
