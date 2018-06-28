import pandas as pd
import time
import gc
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler as SS
import itertools
import pickle
from copy import deepcopy
from collections import Counter
from tqdm import tqdm

from myutils import timer, reduce_mem_usage

categorical = ["user_id","region","city","parent_category_name","category_name","user_type","param_1","param_2","param_3"]
categorical_ex = categorical + ['param_123', 'weekofday']
aggfeats = ['region', 'city', 'parent_category_name', 'category_name', 'user_type', 'param_1', 'param_123']
nonaggfeats = list(set(categorical_ex) - set(aggfeats))

lentrain = 1503424
lentest = 508438
lentrainactive = 14129821
lentestactive = 12824068

def count(df, group_cols, suffix='numcount', agg_type='uint32'):
    aggname = '_'.join(group_cols) + '_' + suffix
    gp = df[group_cols].groupby(group_cols).size().rename(aggname).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    df[aggname] = df[aggname].astype(agg_type)
    del gp; gc.collect()
    return df

def scale_standard(df, ignorecols=[]):
    for col in df.columns:
        if not col in ignorecols:
            df[col] = (SS().fit_transform(df[col].values.reshape(-1, 1))).flatten()        
    return df

def get_smalldiff(df1, df2):
    assert (df1.columns == df2.columns).all(), 'inputs must have same columns'
    th = np.uint(df1.shape[1] * 0.2)
    criteria = {'mean': np.mean, 'var': np.var, 'median': np.median}
    difflist = []
    tempdiff = pd.DataFrame()
    tempdiff['colname'] = [col for col in df1.columns]
    for k, c in criteria.items():
        tempdiff[k] = [np.abs(c(df1[col]) - c(df2[col])) for col in df1.columns]
    tempdiff = scale_standard(tempdiff, ignorecols=['colname'])
    sums = np.zeros(tempdiff.shape[0])
    for key in criteria.keys():
        sums += np.abs(tempdiff[key])
    tempdiff['result'] = sums
    sortdiff = tempdiff.sort_values(by='result')
    return sortdiff['colname'][: th].tolist()

def mynunique(x):
    return len(set(x.values))

if __name__ == '__main__':
    train = pd.read_feather('../features/train/categorical_features_train.feather')
    test = pd.read_feather('../features/test/categorical_features_test.feather')
    train_active = pd.read_feather('../features/train_active/categorical_features_train_active.feather')
    test_active = pd.read_feather('../features/test_active/categorical_features_test_active.feather')
    print(train.shape)
    print(test.shape)
    print(train_active.shape)
    print(test_active.shape)

    df = pd.concat([train, test, train_active, test_active])
    df.drop(nonaggfeats, axis=1, inplace=True)
    print(df.shape)

    support_df = pd.DataFrame()
    for i in range(1, 2):
        for comb in tqdm(list(itertools.combinations(aggfeats, i))):
            group_cols = list(comb)
            target_cols = [col for col in aggfeats if not col in group_cols]
            aggnames = {target: '-'.join(group_cols) + target + '_numunique' for target in target_cols}
            aggfuncs = {target: mynunique for target in target_cols}
            #gp = df[aggfeats].groupby(group_cols).agg(aggfuncs).reset_index().rename(columns=aggnames)
            gp = df[aggfeats].groupby(group_cols).agg(aggfuncs).reset_index().rename(columns=aggnames)

            for aggname in aggnames.values():
                support_df[aggname] = gp[aggname]
                support_df[aggname] = gp[aggname].fillna(gp[aggname].max())
                support_df[aggname] = gp[aggname].astype('uint32')
            del gp; gc.collect()

    df = support_df
    del support_df; gc.collect()
    print(df.shape)

    train = df[:lentrain]
    test = df[lentrain:lentrain+lentest]
    train_active = df[lentrain+lentest: lentrain+lentest+lentrainactive]
    test_active = df[lentrain+lentest+lentrainactive: lentrain+lentest+lentrainactive+lentestactive]
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    train_active.reset_index(drop=True, inplace=True)
    test_active.reset_index(drop=True, inplace=True)
    print(train.shape)
    print(test.shape)
    print(train_active.shape)
    print(test_active.shape)

    train = np.log1p(train) 
    train = scale_standard(train)
    test = np.log1p(test)
    test = scale_standard(test)
    train_active = np.log1p(train_active) 
    train_active = scale_standard(train_active)
    test_active = np.log1p(test_active)
    test_active = scale_standard(test_active)
    print(train.shape)
    print(test.shape)
    print(train_active.shape)
    print(test_active.shape)

    res = get_smalldiff(train, test)
    train = train[res]
    test = test[res]
    train_active = train_active[res]
    test_active = test_active[res]
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    train_active.reset_index(drop=True, inplace=True)
    test_active.reset_index(drop=True, inplace=True)
    print(train.shape)
    print(test.shape)
    print(train_active.shape)
    print(test_active.shape)

    train.to_feather('../features/train/Agg_Unique_Silver_train.feather')
    test.to_feather('../features/test/Agg_Unique_Silver_test.feather')
    train_active.to_feather('../features/train_active/Agg_Unique_Silver_train_active.feather')
    test_active.to_feather('../features/test_active/Agg_Unique_Silver_test_active.feather')

    print('done')