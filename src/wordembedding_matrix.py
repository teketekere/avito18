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

from myutils import reduce_mem_usage

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

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

if __name__ == '__main__':

    EMBEDDING_DIM1 = 300
    EMBEDDING_FILE1 = './wordembedding/wiki.ru.vec'
    EMBEDDING_FILE2 = './wordembedding/cc.ru.300.vec'

    with open('./whole_tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)

    embeddings_index1 = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE1))

    vocab_size = len(tokenizer.word_index)+2
    embedding_matrix1 = np.zeros((vocab_size, EMBEDDING_DIM1))
    print(embedding_matrix1.shape)
    # Creating Embedding matrix 
    c = 0 
    c1 = 0 
    w_Y = []
    w_No = []
    for word, i in tokenizer.word_index.items():
        if word in embeddings_index1:
            c +=1
            embedding_vector = embeddings_index1[word]
            w_Y.append(word)
        else:
            embedding_vector = None
            w_No.append(word)
            c1 +=1
        if embedding_vector is not None:    
            embedding_matrix1[i] = embedding_vector

    print(c,c1, len(w_No), len(w_Y))
    print(embedding_matrix1.shape)
    del embeddings_index1
    gc.collect()

    embedding_matrix1 = embedding_matrix1.astype(np.float32)
    #with open('./wordembedding/WordEmbeddingMatrix_wiki.pickle', 'wb') as f:
    #    np.save(f, embedding_matrix1)
    np.save('./wordembedding/WordEmbeddingMatrix_wiki.npy', embedding_matrix1)

    print(" FAST TEXT wiki DONE")

    embeddings_index2 = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE2))

    vocab_size = len(tokenizer.word_index)+2
    embedding_matrix2 = np.zeros((vocab_size, EMBEDDING_DIM1))
    print(embedding_matrix2.shape)
    # Creating Embedding matrix 
    c = 0 
    c1 = 0 
    w_Y = []
    w_No = []
    for word, i in tokenizer.word_index.items():
        if word in embeddings_index2:
            c +=1
            embedding_vector = embeddings_index2[word]
            w_Y.append(word)
        else:
            embedding_vector = None
            w_No.append(word)
            c1 +=1
        if embedding_vector is not None:    
            embedding_matrix2[i] = embedding_vector

    print(c,c1, len(w_No), len(w_Y))
    print(embedding_matrix2.shape)
    del embeddings_index2
    gc.collect()

    embedding_matrix2 = embedding_matrix2.astype(np.float32)
    #with open('./wordembedding/WordEmbeddingMatrix_cc.pickle', 'wb') as f:
    #    np.save(f, embedding_matrix2)
    np.save('./wordembedding/WordEmbeddingMatrix_cc.npy', embedding_matrix2)

    print(" FAST TEXT CC DONE")
