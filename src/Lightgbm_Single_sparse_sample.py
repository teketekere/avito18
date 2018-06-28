import pandas as pd
import numpy as np
import gc
import os
import pickle
from datetime import datetime as dt

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack, csr_matrix

'''
categorical = ['region',
 'city',
 'parent_category_name',
 'category_name',
 'param_1',
 'user_type',
 'param_123'
 ]
'''
categorical = ["user_id","region","city","parent_category_name","category_name","user_type","image_top_1","param_1","param_2","param_3", "item_seq_number"]
def rmse(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power((y - y0), 2)))

def get_lgb_model(Dtrain, Dvalid, categorical=[], numiter=1500, isValid=True, randomseed=0):
    lgbm_params = {'task': 'train',
                    'boosting_type': 'gbdt',
                    'objective': 'regression',
                    'metric': 'rmse',
                    #'max_depth': 15,
                    'num_leaves': 300,
                    'feature_fraction': 0.5,
                    'bagging_fraction': 0.75,
                    'bagging_freq': 2,
                    'learning_rate': 0.016,
                    'max_bin': 511,
                    'seed': randomseed
                  }
    evals_results = {}
    # Train lgb
    if isValid == True:
        lgb_clf = lgb.train(lgbm_params,
                            Dtrain,
                            valid_sets=[Dvalid],
                            valid_names=['valid'],
                            num_boost_round=numiter,
                            early_stopping_rounds=50,
                            verbose_eval=50,
                            evals_result=evals_results,
                            feval=None,
                            categorical_feature=categorical
                           )
    else:
        lgb_clf = lgb.train(lgbm_params,
                            Dtrain,
                            valid_sets=[Dvalid],
                            valid_names=['valid'],
                            num_boost_round=numiter,
                            verbose_eval=100,
                            feval=None,
                            categorical_feature=categorical
                           )

    best_iteration = lgb_clf.best_iteration
    print(f'best iteration is {best_iteration}')
    if isValid:
        best_score = evals_results['valid']['rmse'][lgb_clf.best_iteration-1]
        print(f'best score is {best_score}')
    return lgb_clf, best_iteration

def get_lgb_train(trainX, trainy, validX, validy, predictors, categorical):
    Dtrain = lgb.Dataset(trainX.values, label=trainy.values.ravel(), feature_name=predictors, categorical_feature=categorical)
    Dvalid = lgb.Dataset(validX.values, label=validy.values.ravel(), feature_name=predictors, categorical_feature=categorical)
    return Dtrain, Dvalid

if __name__ == '__main__':
    # Example For KFold
    EnableSubmit = True

    # WHOLE
    numiter = 24
    print('WHOLE TRAIN: ', numiter)

    # DODODDODO
    train = pd.read_feather('../features/featured/train_full_4.feather')
    y = pd.read_csv('../input/train.csv', usecols=['deal_probability'])
    lentrain = train.shape[0]

    # TFIDF for merge
    with open('./tfidf/kernel_tfidf.pickle', 'rb') as f:
        tfidf_feature = pickle.load(f)
    dummy_tfvocab = ['dm_'+str(i) for i in range(tfidf_feature.shape[1])]
    predictors = list(train.columns) + list(dummy_tfvocab)    
    train_tfidf = tfidf_feature[:lentrain]
    test_tfidf = tfidf_feature[lentrain:]
    del tfidf_feature; gc.collect

    train = hstack([csr_matrix(train), train_tfidf])
    print(train.shape)
    print(y.shape)

    randomseeds = [34, 8742, 902, 424, 4891]
    preds = pd.DataFrame()
    for seed in randomseeds:
        #Dtrain, Dvalid = get_lgb_train(train, y, train.iloc[0:2, :], y.iloc[0:2, :], predictors, categorical)
        Dtrain = lgb.Dataset(train, label=y.values.ravel(), feature_name=predictors, categorical_feature=categorical)
        Dvalid = lgb.Dataset(train.tocsr()[:2].tocsc(), label=y.iloc[:2, :].values.ravel(), feature_name=predictors, categorical_feature=categorical)

        del train, y; gc.collect()
        lgb_model, best_iter = get_lgb_model(Dtrain, Dvalid, categorical=categorical, numiter=numiter, isValid=False, randomseed=seed)
        print(best_iter)
        del Dtrain, Dvalid; gc.collect()

        test = pd.read_feather('../features/featured/test_full_4.feather')
        test = hstack([csr_matrix(test), test_tfidf])
        print(test.shape)
        preds[str(seed)] = lgb_model.predict(test)
        del test ; gc.collect()

        # DODODDODO
        train = pd.read_feather('../features/featured/train_full_4.feather')
        y = pd.read_csv('../input/train.csv', usecols=['deal_probability'])
        train = hstack([csr_matrix(train), train_tfidf])

    del train, y; gc.collect()
    test = pd.read_feather('../features/featured/test_full_4.feather')
    test = hstack([csr_matrix(test), test_tfidf])
    datetime = dt.now().strftime('%Y_%m%d_%H%M_%S')
    fileprefix = '../subs/lightgbm_single_ensemble_'
    if EnableSubmit == True:
        subs = pd.read_csv('../input/test.csv', usecols=['item_id'])
        pred_ensemble = np.zeros(test.shape[0])
        for col in preds.columns:
            pred_ensemble += preds[col]
        pred_ensemble = pred_ensemble / np.float(len(randomseeds))
        pred_ensemble = np.clip(pred_ensemble, 0, 1)
        subs['deal_probability'] = pred_ensemble
        filename = fileprefix+datetime+'_linear.csv.gz'
        subs.to_csv(filename, index=False, float_format='%.9f', compression='gzip')
        print(subs.shape)
        del subs; gc.collect()    

        subs = pd.read_csv('../input/test.csv', usecols=['item_id'])
        pred_ensemble = np.zeros(test.shape[0])
        for col in preds.columns:
            preds[col] = np.clip(preds[col], 0.0000001, 1)
            pred_ensemble += np.log(preds[col]) / np.float(len(randomseeds))
        pred_ensemble = np.exp(pred_ensemble)
        pred_ensemble = np.clip(pred_ensemble, 0, 1)
        subs['deal_probability'] = pred_ensemble
        filename = fileprefix+datetime+'_exp.csv.gz'
        subs.to_csv(filename, index=False, float_format='%.9f', compression='gzip')
        print(subs.shape)
        del subs; gc.collect()    
