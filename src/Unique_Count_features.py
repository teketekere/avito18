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
    with open('./Golden_Uniqueount_aggs.pickle', 'rb') as f:
        count_agg_list = pickle.load(f)
    
    print(count_agg_list)
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

    for cols in tqdm(count_agg_list):
        group_cols = cols[: -1]
        target = cols[-1]
        print(group_cols, target, cols)
        aggname = '-'.join(cols) + '_numunique'
        gp = df[cols].groupby(group_cols)[target].nunique().rename(aggname).to_frame().reset_index()
        df = df.merge(gp, on=group_cols, how='left')
        df[aggname] = df[aggname].fillna(df[aggname].max())
        df[aggname] = df[aggname].astype('uint32')
        del gp; gc.collect()

    print(df.shape)
    print(df.nunique())

    df = reduce_mem_usage(df)
    df.drop(aggfeats, axis=1, inplace=True)
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

    train.to_feather('../features/train/Agg_numunique_Golden_features_train.feather')
    test.to_feather('../features/test/Agg_numunique_Golden_features_test.feather')
    train_active.to_feather('../features/train_active/Agg_numunique_Golden_features_train_active.feather')
    test_active.to_feather('../features/test_active/Agg_numunique_Golden_features_test_active.feather')

    print('done')