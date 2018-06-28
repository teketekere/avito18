import pandas as pd 
import numpy as np 
import time 
import gc 
import pickle

np.random.seed(42)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from keras.models import Model
from keras.layers import Input, Dropout, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, BatchNormalization, Conv1D, MaxPooling1D, Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import backend as K
from keras.models import Model

from keras import optimizers

from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

import threading
import multiprocessing
from multiprocessing import Pool, cpu_count
from contextlib import closing

from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

lentrain = 1503424
lentest = 508438
lentrainactive = 14129821
lentestactive = 12824068
textfeats = ['title', 'description']
max_seq_title_description_length = 100
max_words_title_description = 400000

### rmse loss for keras
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred))) 

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
    train_active = pd.read_csv('../input/train_active.csv', usecols=textfeats)
    test_active = pd.read_csv('../input/test_active.csv', usecols=textfeats)

    df = pd.concat([train_active, test_active])
    del train, test, train_active, test_active
    gc.collect()
    print(df.shape)

    df['title_description'] = (df['title']+" "+df['description']).astype(str)
    df.drop(textfeats, axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    gc.collect()
    print(df.shape)

    train_active = df.iloc[:lentrainactive, :]
    test_active = df.iloc[lentrainactive: lentrainactive+lentestactive, :]

    del df
    gc.collect()

    with open('./title_description_param_tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    train_active.reset_index(drop=True, inplace=True)
    test_active.reset_index(drop=True, inplace=True)

    train_active['seq_title_description'] = tokenizer.texts_to_sequences(train_active['title_description'].str.lower())
    test_active['seq_title_description'] = tokenizer.texts_to_sequences(test_active['title_description'].str.lower())

    train_active.drop('title_description', axis=1, inplace=True)
    test_active.drop('title_description', axis=1, inplace=True)
    gc.collect()


    print(train_active.shape)
    print(test_active.shape)

    train_active.to_csv('../features/train_active/seq_title_description_train_active.csv', index=False)
    test_active.to_csv('../features/test_active/seq_title_description_test_active.csv', index=False)
