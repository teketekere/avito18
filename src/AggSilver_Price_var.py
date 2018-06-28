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
aggfeats = ['region', 'city', 'parent_category_name', 'category_name', 'user_type', 'weekofday', 'param_1', 'param_123']
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
   
if __name__ == '__main__':
    target = 'price'

    train = pd.read_feather('../features/train/categorical_features_train.feather')
    test = pd.read_feather('../features/test/categorical_features_test.feather')
    trainp = pd.read_csv('../input/train.csv', usecols=[target])
    testp = pd.read_csv('../input/test.csv', usecols=[target])
    trainp.fillna(trainp.mean(), inplace=True)
    testp.fillna(testp.mean(), inplace=True)
    train = pd.concat([train, trainp], axis=1)
    test = pd.concat([test, testp], axis=1)
    del trainp, testp; gc.collect()
        
    train_active = pd.read_feather('../features/train_active/categorical_features_train_active.feather')
    test_active = pd.read_feather('../features/test_active/categorical_features_test_active.feather')
    trainap = pd.read_csv('../input/train_active.csv', usecols=[target])
    testap = pd.read_csv('../input/test_active.csv', usecols=[target])
    trainap.fillna(trainap.mean(), inplace=True)
    testap.fillna(testap.mean(), inplace=True)
    train_active = pd.concat([train_active, trainap], axis=1)
    test_active = pd.concat([test_active, testap], axis=1)
    del trainap, testap; gc.collect()

    print(train.shape)
    print(test.shape)
    print(train_active.shape)
    print(test_active.shape)

    df = pd.concat([train, test, train_active, test_active])
    df.drop(nonaggfeats, axis=1, inplace=True)
    print(df.shape)

    for i in range(1, 5):
        for comb in tqdm(list(itertools.combinations(aggfeats, i))):
            group_cols = list(comb)        
            print(group_cols)
            aggname = '-'.join(group_cols+[target]) + '-var'
            gp = df[group_cols+[target]].groupby(group_cols)[target].var().rename(aggname).to_frame().reset_index()
            df = df.merge(gp, on=group_cols, how='left')
            df[aggname] = df[aggname].fillna(df[aggname].mean())
            df[aggname] = df[aggname].astype('float32')
            del gp; gc.collect()

    df.drop(aggfeats+[target], axis=1, inplace=True)

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

    train = np.log(train+0.001) 
    train = scale_standard(train)
    test = np.log(test+0.001)
    test = scale_standard(test)
    train_active = np.log(train_active+0.001) 
    train_active = scale_standard(train_active)
    test_active = np.log(test_active+0.001)
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

    train.to_feather('../features/train/Agg_Price_var_Silver_train.feather')
    test.to_feather('../features/test/Agg_Price_var_Silver_test.feather')
    train_active.to_feather('../features/train_active/Agg_Price_var_Silver_train_active.feather')
    test_active.to_feather('../features/test_active/Agg_Price_var_Silver_test_active.feather')

    print('done')