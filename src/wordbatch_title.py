import numpy as np
import pandas as pd
import os
import gc
import re
import string
import time
import matplotlib.pyplot as plt
import pickle

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix

import nltk
from nltk.corpus import stopwords 
import wordbatch
from wordbatch.extractors import WordBag
from wordbatch.models import FM_FTRL


#nltk.download('stopwords')
lentrain = 1503424
lentest = 508438
lentrainactive = 14129821
lentestactive = 12824068

#stopwords_kernel = {x: 1 for x in stopwords.words('russian')}
stopwords_kernel = list(set(stopwords.words('russian')))
non_alphanums = re.compile(u'[^A-Za-z0-9]+')
non_alphanumpunct = re.compile(u'[^A-Za-z0-9\.?!,; \(\)\[\]\'\"\$]+')
RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])
stop_words = list(set(stopwords.words('russian')))
russian_stop = set(stopwords.words('russian'))
punctuation = string.punctuation

textfeats = ['title', 'description', 'param_1', 'param_2', 'param_3']

def normalize_text(text):
    text = text.lower().strip()
    for s in string.punctuation:
        text = text.replace(s, ' ')
    text = text.strip().split(' ')
    return u' '.join(x for x in text if len(x) > 1 and x not in stopwords_kernel)

def rmse(predicted, actual):
    return np.sqrt(((predicted - actual) ** 2).mean())

def cleanName(text):
    try:
        textProc = text.lower()
        # textProc = " ".join(map(str.strip, re.split('(\d+)',textProc)))
        #regex = re.compile(u'[^[:alpha:]]')
        #textProc = regex.sub(" ", textProc)
        textProc = re.sub('[!@#$_“”¨«»®´·º½¾¿¡§£₤‘’]', '', textProc)
        textProc = " ".join(textProc.split())
        return textProc
    except: 
        return "name error"

if __name__ == '__main__':
    # Wordbatch
    train = pd.read_csv('../input/train.csv', usecols=textfeats)
    test = pd.read_csv('../input/test.csv', usecols=textfeats)
    train_active = pd.read_csv('../input/train_active.csv', usecols=textfeats)
    test_active = pd.read_csv('../input/test_active.csv', usecols=textfeats)

    '''
    lentrain = train.shape[0]
    lentest = test.shape[0]
    lentrainactive = train_active.shape[0]
    lentestactive = test_active.shape[0]
    '''

    df = pd.concat([train, test, train_active, test_active])
    del train, test, train_active, test_active
    gc.collect()
    print(df.shape)

    for col in textfeats:
        df[col] = df[col].astype(str)
        df[col] = df[col].fillna('nothing')
        df[col] = df[col].str.lower()
        df[col] = df[col].apply(lambda x: cleanName(x))

    df['title_description'] = (df['title']+" "+df['description']).astype(str)
    df['title_param'] = (df['title']+' '+df['param_1']+' '+df['param_2']+' '+df['param_3']).astype(str)
    df.drop(textfeats, axis=1, inplace=True)
    gc.collect()
    print(df.shape)

    training = df.iloc[:lentrain, :]
    test  = df.iloc[lentrain:lentrain+lentest, :]
    train_active = df.iloc[lentrain+lentest:lentrain+lentest+lentrainactive, :]
    test_active = df.iloc[lentrain+lentest+lentrainactive: lentrain+lentest+lentrainactive+lentestactive]

    testing = pd.concat([test, train_active ,test_active])
    del test, train_active, test_active
    gc.collect()

    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                                "hash_ngrams_weights": [1.5, 1.0],
                                                                "hash_size": 2 ** 29,
                                                                "norm": None,
                                                                "tf": 'binary',
                                                                "idf": None,
                                                                }), procs=24)
    wb.dictionary_freeze = True
    X_name_train = wb.fit_transform(train['title_param'])
    print(X_name_train.shape)
    X_name_test = wb.transform(testing['title_param'])
    print(X_name_test.shape)
    del(wb)
    gc.collect()

    mask = np.where(X_name_train.getnnz(axis=0) > 3)[0]
    X_name_train = X_name_train[:, mask]
    print(X_name_train.shape)
    X_name_test = X_name_test[:, mask]
    print(X_name_test.shape)

    training.drop('title_param', axis=1, inplace=True)
    testing.drop('title_param', axis=1, inplace=True)
    gc.collect()

    y = pd.read_csv('../input/train.csv', usecols=['deal_probability'])
    X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_name_train,
                                                                y,
                                                                test_size = 0.5,
                                                                shuffle = False)
    model = Ridge(solver="sag", fit_intercept=True, random_state=42, alpha=5)
    model.fit(X_train_1, y_train_1)
    train_ridge = model.predict(X_name_train)
    test_ridge = model.predict(X_name_test)
    print(rmse(model.predict(X_train_2), y_train_2))

    model = Ridge(solver="sag", fit_intercept=True, random_state=4882, alpha=5)
    model.fit(X_train_2, y_train_2)
    train_ridge += model.predict(X_name_train)
    test_ridge += model.predict(X_name_test)
    print(rmse(model.predict(X_train_1), y_train_1))

    train_ridge /= 2.0
    test_ridge /= 2.0

    del X_train_1, X_train_2, y_train_1, y_train_2
    gc.collect()

    train_ridgedf = pd.DataFrame()
    train_ridgedf['wordbach_title_ridge'] = train_ridge.iloc[:lentrain, :].flatten()
    train_active_ridgedf = pd.DataFrame()
    train_active_ridgedf['wordbach_title_ridge'] = train_ridge.iloc[lentrain:, :].flatten()

    test_ridgedf = pd.DataFrame()
    test_ridgedf['wordbach_title_ridge'] = test_ridge.iloc[:lentest, :].flatten()
    test_active_ridgedf = pd.DataFrame()
    test_active_ridgedf['wordbach_title_ridge'] = test_ridge.iloc[lentest:, :].flatten()

    print(train_ridgedf.shape)
    print(train_active_ridgedf.shape)
    print(test_ridgedf.shape)
    print(test_active_ridgedf.shape)

    train_ridgedf.reset_index(drop=True, inplace=True)
    train_active_ridgedf.reset_index(drop=True, inplace=True)
    test_ridgedf.reset_index(drop=True, inplace=True)
    test_active_ridgedf.reset_index(drop=True, inplace=True)

    train_ridgedf.to_feather('../features/train/wordbatch_title_ridge_train.feather')
    test_ridgedf.to_feather('../features/test/wordbatch_title_ridge_test.feather')
    train_active_ridgedf.to_feather('../features/train_active/wordbatch_title_ridge_train_active.feather')
    test_active_ridgedf.to_feather('../features/test_active/wordbatch_title_ridge_test_active.feather')

    del train_ridgedf, test_ridgedf, train_active_ridgedf, test_active_ridgedf, train_ridge, test_ridge
    gc.collect()

    print('title tsvd')
    n_comp = 10
    tsvd = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    tsvd.fit(X_name_train)

    train_svd = pd.DataFrame(tsvd.transform(X_name_train))
    test_svd = pd.DataFrame(tsvd.transform(X_name_test))
    train_svd.columns = ['svd_wordbatch_title_'+str(i+1) for i in range(n_comp)]
    test_svd.columns =  ['svd_wordbatch_title_'+str(i+1) for i in range(n_comp)]

    print(train_svd.shape)
    print(test_svd.shape)

    train_svd_orig = train_svd.iloc[:lentrain, :]
    train_svd_active = train_svd.iloc[lentrain:, :]
    test_svd_orig = test_svd.iloc[:lentest, :]
    test_svd_active = test_svd.iloc[lentest:, :]
    
    train_svd_orig.reset_index(drop=True, inplace=True)
    test_svd_orig.reset_index(drop=True, inplace=True)
    train_svd_active.reset_index(drop=True, inplace=True)
    test_svd_active.reset_index(drop=True, inplace=True)

    print(train_svd_orig.shape)
    print(test_svd_orig.shape)
    print(train_svd_active.shape)
    print(test_svd_active.shape)

    train_svd_orig.to_feather('../features/train/wordbatch_title_tsvd_train.feather')
    test_svd_orig.to_feather('../features/test/wordbatch_title_tsvd_test.feather')
    train_svd_active.to_feather('../features/train_active/wordbatch_title_tsvd_train_active.feather')
    test_svd_active.to_feather('../features/test_active/wordbatch_title_tsvd_test_active.feather')

    del train_svd_orig, test_svd_orig, train_svd_active, test_svd_active, train_svd, test_svd
    gc.collect()

    # Description
    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                                "hash_ngrams_weights": [1.0, 1.0],
                                                                "hash_size": 2 ** 28,
                                                                "norm": "l2",
                                                                "tf": 1.0,
                                                                "idf": None,
                                                                }), procs=24)
    wb.dictionary_freeze = True
    X_desc_train = wb.fit_transform(training['title_description'])
    print(X_desc_train.shape)
    X_desc_test = wb.transform(testing['title_description'])
    print(X_desc_test.shape)
    del(wb)
    gc.collect()

    mask = np.where(X_desc_train.getnnz(axis=0) > 3)[0]
    X_desc_train = X_desc_train[:, mask]
    print(X_desc_train.shape)
    X_desc_test = X_desc_test[:, mask]
    print(X_desc_test.shape)

    del training, testing
    gc.collect()

    X_train_1, X_train_2, y_train_1, y_train_2 = train_test_split(X_desc_train,
                                                                y,
                                                                test_size = 0.5,
                                                                shuffle = False)
    model = Ridge(solver="sag", fit_intercept=True, random_state=42, alpha=5)
    model.fit(X_train_1, y_train_1)
    train_ridge = model.predict(X_desc_train)
    test_ridge = model.predict(X_desc_test)
    print(rmse(model.predict(X_train_2), y_train_2))

    model = Ridge(solver="sag", fit_intercept=True, random_state=4882, alpha=5)
    model.fit(X_train_2, y_train_2)
    train_ridge += model.predict(X_desc_train)
    test_ridge += model.predict(X_desc_test)
    print(rmse(model.predict(X_train_1), y_train_1))

    train_ridge /= 2.0
    test_ridge /= 2.0

    del X_train_1, X_train_2, y_train_1, y_train_2
    gc.collect()

    train_ridgedf = pd.DataFrame()
    train_ridgedf['wordbach_description_ridge'] = train_ridge.iloc[:lentrain, :].flatten()
    train_active_ridgedf = pd.DataFrame()
    train_active_ridgedf['wordbach_description_ridge'] = train_ridge.iloc[lentrain:, :].flatten()

    test_ridgedf = pd.DataFrame()
    test_ridgedf['wordbach_description_ridge'] = test_ridge.iloc[:lentest, :].flatten()
    test_active_ridgedf = pd.DataFrame()
    test_active_ridgedf['wordbach_description_ridge'] = test_ridge.iloc[lentest:, :].flatten()

    print(train_ridgedf.shape)
    print(train_active_ridgedf.shape)
    print(test_ridgedf.shape)
    print(test_active_ridgedf.shape)

    train_ridgedf.reset_index(drop=True, inplace=True)
    train_active_ridgedf.reset_index(drop=True, inplace=True)
    test_ridgedf.reset_index(drop=True, inplace=True)
    test_active_ridgedf.reset_index(drop=True, inplace=True)

    train_ridgedf.to_feather('../features/train/wordbatch_description_ridge_train.feather')
    test_ridgedf.to_feather('../features/test/wordbatch_description_ridge_test.feather')
    train_active_ridgedf.to_feather('../features/train_active/wordbatch_description_ridge_train_active.feather')
    test_active_ridgedf.to_feather('../features/test_active/wordbatch_description_ridge_test_active.feather')

    del train_ridgedf, test_ridgedf, train_active_ridgedf, test_active_ridgedf, train_ridge, test_ridge
    gc.collect()

    print('title tsvd')
    tsvd = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    tsvd.fit(X_desc_train)

    train_svd = pd.DataFrame(tsvd.transform(X_desc_train))
    test_svd = pd.DataFrame(tsvd.transform(X_desc_test))
    train_svd.columns = ['svd_wordbatch_description_'+str(i+1) for i in range(n_comp)]
    test_svd.columns =  ['svd_wordbatch_description_'+str(i+1) for i in range(n_comp)]

    print(train_svd.shape)
    print(test_svd.shape)

    train_svd_orig = train_svd.iloc[:lentrain, :]
    train_svd_active = train_svd.iloc[lentrain:, :]
    test_svd_orig = test_svd.iloc[:lentest, :]
    test_svd_active = test_svd.iloc[lentest:, :]
    
    train_svd_orig.reset_index(drop=True, inplace=True)
    test_svd_orig.reset_index(drop=True, inplace=True)
    train_svd_active.reset_index(drop=True, inplace=True)
    test_svd_active.reset_index(drop=True, inplace=True)

    print(train_svd_orig.shape)
    print(test_svd_orig.shape)
    print(train_svd_active.shape)
    print(test_svd_active.shape)

    train_svd_orig.to_feather('../features/train/wordbatch_description_tsvd_train.feather')
    test_svd_orig.to_feather('../features/test/wordbatch_description_tsvd_test.feather')
    train_svd_active.to_feather('../features/train_active/wordbatch_description_tsvd_train_active.feather')
    test_svd_active.to_feather('../features/test_active/wordbatch_description_tsvd_test_active.feather')

    del train_svd_orig, test_svd_orig, train_svd_active, test_svd_active, train_svd, test_svd
    gc.collect()

    # Ensemble
    dummy_cols = ['parent_category_name', 'category_name', 'user_type',
                'region', 'city']
    df_train = pd.read_csv('../input/train.csv', usecols=dummy_cols)
    df_test  = pd.read_csv('../input/test.csv' , usecols=dummy_cols)
    y_train = pd.read_csv('../input/train.csv', usecols=['deal_probability'])

    sparse_merge_train = hstack((X_name_train, X_desc_train)).tocsr()
    sparse_merge_test = hstack((X_name_test, X_desc_test)).tocsr()
    print(sparse_merge_train.shape)
    for col in dummy_cols:
        print(col)
        lb = LabelBinarizer(sparse_output=True)
        sparse_merge_train = hstack((sparse_merge_train, lb.fit_transform(df_train[[col]].fillna('')))).tocsr()
        print(sparse_merge_train.shape)
        sparse_merge_test = hstack((sparse_merge_test, lb.transform(df_test[[col]].fillna('')))).tocsr()

    del X_desc_test, X_name_test
    del X_desc_train, X_name_train, lb, df_train, df_test
    gc.collect()    