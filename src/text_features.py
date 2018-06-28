import numpy as np
import pandas as pd
import os
import gc
import re
import string
import time
import pickle

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import hstack, csr_matrix

import nltk
from nltk.corpus import stopwords 


nltk.download('stopwords')
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
    train = pd.read_csv('../input/train.csv', usecols=['description', 'title'])
    test = pd.read_csv('../input/test.csv', usecols=['description', 'title'])
    train_active = pd.read_csv('../input/train_active.csv', usecols=['description', 'title'])
    test_active = pd.read_csv('../input/test_active.csv', usecols=['description', 'title'])

    print(train.shape)
    print(test.shape)
    print(train_active.shape)
    print(test_active.shape)

    df = pd.concat([train, test, train_active, test_active])
    del train, test, train_active, test_active
    gc.collect()

    for col in textfeats:
        df[col] = df[col].astype(str)
        df[col] = df[col].fillna('missing')
        df[col + '_titleword_count'] = df[col].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
        df[col + '_upper_case_word_count'] = df[col].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

        df[col] = df[col].str.lower()
        df[col + '_num_stopwords'] = df[col].apply(lambda x: len([wrd for wrd in x.split() if wrd.lower() in stop_words]))
        df[col + '_num_punctuations'] = df[col].apply(lambda x: len("".join(_ for _ in x if _ in punctuation)))
        df[col + '_num_alphabets'] = df[col].apply(lambda comment: len([c for c in comment if c.isupper()]))
        df[col + '_num_digits'] = df[col].apply(lambda comment: (comment.count('[0-9]')))
        df[col + '_num_chars'] = df[col].apply(len) # Count number of Characters
        df[col + '_num_words'] = df[col].apply(lambda comment: len(comment.split())) # Count number of Words
        df[col + '_num_unique_words'] = df[col].apply(lambda comment: len(set(w for w in comment.split())))
        df[col + '_chars_by_words'] = df[col + '_num_chars'] / (df[col + '_num_words'] + 1)
        df[col + '_words_by_uniquewords'] = df[col + '_num_unique_words'] / (df[col+'_num_words'] + 1)
        df[col + '_punctuations_by_chars'] = df[col+'_num_punctuations'] / (df[col + '_num_chars'] + 1)
        df[col + '_punctuations_by_words'] = df[col + '_num_punctuations'] / (df[col + '_num_words'] + 1)
        df[col + '_digits_by_chars'] = df[col + '_num_digits'] / (df[col + '_num_chars'] + 1)
        df[col + '_alphabets_by_chars'] = df[col + '_num_alphabets'] / (df[col + '_num_chars'] + 1)
        df[col + '_stopwords_by_words'] = df[col + '_num_stopwords'] / (df[col + '_num_words'] + 1)
        df[col + '_mean'] = df[col].apply(lambda x: 0 if len(x) == 0 else float(len(x.split())) / len(x)) * 10
    print(df.columns)
    df['title_description_len_ratio'] = (df['title_num_chars'].astype(np.float)) / (df['description_num_chars'].astype(np.float) + 1)
    df.head()

    df = df.drop(['title', 'description'], axis=1)

    train_active = df[lentrain+lentest:lentrain+lentest+lentrainactive]
    test_active = df[lentrain+lentest+lentrainactive: lentrain+lentest+lentrainactive+lentestactive]
    train_active.reset_index(drop=True, inplace=True)
    test_active.reset_index(drop=True, inplace=True)
    print(train_active.shape)
    print(test_active.shape)

    train_active.to_feather('../features/train_active/textfeatures_train_active.feather')
    test_active.to_feather('../features/test_active/textfeatures_test_active.feather')