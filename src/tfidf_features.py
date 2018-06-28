import numpy as np
import pandas as pd
import os
import gc
import re
import string

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix

import nltk
from nltk.corpus import stopwords 
nltk.download('stopwords')

lentrain = 1503424
lentest = 508438
lentrainactive = 14129821
lentestactive = 12824068
russian_stop = set(stopwords.words('russian'))
textfeats = ['title', 'description']
#textfeats = ['title', 'description', 'param_1', 'param_2', 'param_3']

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

def get_col(col_name):
    return lambda x: x[col_name]

if __name__ == '__main__':
    print('start')
    train = pd.read_csv('../input/train.csv', usecols=textfeats)
    test = pd.read_csv('../input/test.csv', usecols=textfeats)
    train_active = pd.read_csv('../input/train_active.csv', usecols=textfeats)
    test_active = pd.read_csv('../input/test_active.csv', usecols=textfeats)

    df = pd.concat([train, test, train_active, test_active])
    del train, test, train_active, test_active
    gc.collect()
    print(df.shape)

    for col in textfeats:
        df[col] = df[col].fillna('nothing')
        df[col] = df[col].apply(lambda x: cleanName(x))
        print(df.shape)

    tfidf_para = {
        "stop_words": russian_stop,
        "analyzer": 'word',
        "token_pattern": r'\w{1,}',
        "sublinear_tf": True,
        "dtype": np.float32,
        "norm": 'l2',
        "smooth_idf":False
    }

    vectorizer = FeatureUnion([
            ('description',TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=17000,
                **tfidf_para,
                preprocessor=get_col('description'))),
            ('title',CountVectorizer(
                ngram_range=(1, 2),
                stop_words = russian_stop,
                preprocessor=get_col('title')))
        ])

    vectorizer.fit(df.to_dict('records'))
    ready_df = vectorizer.transform(df.to_dict('records'))
    print(ready_df.shape)
    del df
    gc.collect()

    train_df = ready_df[:lentrain]
    y = pd.read_csv('../input/train.csv', usecols=['deal_probability'])

    ridge_params = {'alpha':30.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                    'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':786}
    ridge = Ridge(**ridge_params)
    ridge.fit(train_df, y)
    pred_ridge = ridge.predict(ready_df)

    train_ridge = pred_ridge[:lentrain]
    test_ridge = pred_ridge[lentrain:lentrain+lentest]
    train_active_ridge = pred_ridge[lentrain+lentest: lentrain+lentest+lentrainactive]
    test_active_ridge = pred_ridge[lentrain+lentest+lentrainactive: lentrain+lentest+lentrainactive+lentestactive]

    print(train_ridge.shape)
    print(test_ridge.shape)
    print(train_active_ridge.shape)
    print(test_active_ridge)

    train_ridge_df = pd.DataFrame()
    test_ridge_df = pd.DataFrame()
    train_active_ridge_df = pd.DataFrame()
    test_active_ridge_df = pd.DataFrame()

    train_ridge_df['tfidf_td_counttp_ridge'] = pd.Series(train_ridge.flatten())
    test_ridge_df['tfidf_td_counttp_ridge'] = pd.Series(test_ridge.flatten())
    train_active_ridge_df['tfidf_td_counttp_ridge'] = pd.Series(train_active_ridge.flatten())
    test_active_ridge_df['tfidf_td_counttp_ridge'] = pd.Series(test_active_ridge.flatten())

    train_ridge_df.to_feather('../features/train/tfidf_td_counttp_ridge_train.feather')
    test_ridge_df.to_feather('../features/test/tfidf_td_counttp_ridge_test.feather')
    train_active_ridge_df.to_feather('../features/train_active/tfidf_td_counttp_ridge_train_active.feather')
    test_active_ridge_df.to_feather('../features/test_active/tfidf_td_counttp_ridge_test_active.feather')

    n_comp = 10
    tsvd = TruncatedSVD(n_components=n_comp, algorithm='arpack')
    tsvd.fit(train_df)
    pred_tsvd = tsvd.transform(ready_df)

    train_svd = pd.DataFrame(pred_tsvd[:lentrain])
    test_svd = pd.DataFrame(pred_tsvd[lentrain:lentrain+lentest])
    train_active_svd = pd.DataFrame(pred_tsvd[lentrain+lentest: lentrain+lentest+lentrainactive])
    test_active_svd = pd.DataFrame(pred_tsvd[lentrain+lentest+lentrainactive: lentrain+lentest+lentrainactive+lentestactive])

    train_svd.columns = ['svd_tfidf_td_counttp_'+str(i+1) for i in range(n_comp)]
    test_svd.columns =  ['svd_tfidf_td_counttp_'+str(i+1) for i in range(n_comp)]
    train_active_svd.columns = ['svd_tfidf_td_counttp_'+str(i+1) for i in range(n_comp)]
    test_active_svd.columns =  ['svd_tfidf_td_counttp_'+str(i+1) for i in range(n_comp)]


    print(train_svd.shape)
    print(test_svd.shape)
    print(train_active_svd.shape)
    print(test_active_svd.shape)

    train_svd.to_feather('../features/train/tfidf_td_counttp_tsvd_train.feather')
    test_svd.to_feather('../features/test/tfidf_td_counttp_tsvd_test.feather')
    train_active_svd.to_feather('../features/train_active/tfidf_td_counttp_tsvd_train.feather')
    test_active_svd.to_feather('../features/test_active/tfidf_td_counttp_tsvd_test.feather')