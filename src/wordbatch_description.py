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

from myutils import reduce_mem_usage


#nltk.download('stopwords')
lentrain = 1503424
#stopwords_kernel = {x: 1 for x in stopwords.words('russian')}
stopwords_kernel = list(set(stopwords.words('russian')))
non_alphanums = re.compile(u'[^A-Za-z0-9]+')
non_alphanumpunct = re.compile(u'[^A-Za-z0-9\.?!,; \(\)\[\]\'\"\$]+')
RE_PUNCTUATION = '|'.join([re.escape(x) for x in string.punctuation])
stop_words = list(set(stopwords.words('russian')))
russian_stop = set(stopwords.words('russian'))
punctuation = string.punctuation

textfeats = ['title', 'description']

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
    train = pd.read_csv('../input/train.csv', usecols=['description', 'title'])
    test = pd.read_csv('../input/test.csv', usecols=['description', 'title'])

    for col in textfeats:
        train[col] = train[col].astype(str)
        train[col] = train[col].fillna('missing')
        train[col] = train[col].str.lower()
        train[col] = train[col].apply(lambda x: cleanName(x))
    for col in textfeats:
        test[col] = test[col].astype(str)
        test[col] = test[col].fillna('missing')
        test[col] = test[col].str.lower()
        test[col] = test[col].apply(lambda x: cleanName(x))

    train['title_description'] = (train['title']+" "+train['description']).astype(str)
    #train['title_param'] = (train['title']+' '+train['param_1']+' '+train['param_2']+' '+train['param_3']).astype(str)
    test['title_description'] = (test['title']+" "+test['description']).astype(str)
    #test['title_param'] = (test['title']+' '+test['param_1']+' '+test['param_2']+' '+test['param_3']).astype(str)
    print(train.shape)
    print(test.shape)
    wb = wordbatch.WordBatch(normalize_text, extractor=(WordBag, {"hash_ngrams": 2,
                                                                "hash_ngrams_weights": [1.0, 1.0],
                                                                "hash_size": 2 ** 28,
                                                                "norm": "l2",
                                                                "tf": 1.0,
                                                                "idf": None,
                                                                }), procs=8)
    wb.dictionary_freeze = True
    X_desc_train = wb.fit_transform(train['title_description'])
    print(X_desc_train.shape)
    X_desc_test = wb.transform(test['title_description'])
    print(X_desc_test.shape)
    del(wb)
    gc.collect()

    mask = np.where(X_desc_train.getnnz(axis=0) > 3)[0]
    X_desc_train = X_desc_train[:, mask]
    print(X_desc_train.shape)
    X_desc_test = X_desc_test[:, mask]
    print(X_desc_test.shape)

    with open('./wordbatch_description_train.pickle', 'wb') as f:
        pickle.dump(X_desc_train, f)

    with open('./wordbatch_description_test.pickle', 'wb') as f:
        pickle.dump(X_desc_test, f)
        