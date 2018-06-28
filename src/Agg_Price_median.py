import pandas as pd
import time
import gc
import numpy as np

from sklearn.preprocessing import LabelEncoder
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

if __name__ == '__main__':
    with open('./Golden_Price_median_agg.pickle', 'rb') as f:
        agg_list = pickle.load(f)
        
    print(agg_list)
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

    for cols in tqdm(agg_list):
        group_cols = cols[:-1]
        assert target == cols[-1]
        print(group_cols)
        aggname = '-'.join(group_cols+[target]) + '-median'
        gp = df[group_cols+[target]].groupby(group_cols)[target].median().rename(aggname).to_frame().reset_index()
        df = df.merge(gp, on=group_cols, how='left')
        df[aggname] = df[aggname].fillna(df[aggname].mean())
        df[aggname] = df[aggname].astype('float32')
        del gp; gc.collect()

    print(df.shape)
    df.head()

    df = reduce_mem_usage(df)
    df.drop(aggfeats+['price'], axis=1, inplace=True)
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

    train.to_feather('../features/train/Agg_Price_median_Golden_features_train.feather')
    test.to_feather('../features/test/Agg_Price_median_Golden_features_test.feather')
    train_active.to_feather('../features/train_active/Agg_Price_median_Golden_features_train_active.feather')
    test_active.to_feather('../features/test_active/Agg_Price_median_Golden_features_test_active.feather')

    print('done')
    train.nunique()