import pandas as pd
import numpy as np
import gc
import os
import pickle
from datetime import datetime as dt

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
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
                    'num_leaves': 270,
                    'feature_fraction': 0.5,
                    'bagging_fraction': 0.75,
                    'bagging_freq': 2,
                    'learning_rate': 0.016,
                    'max_bin': 255,
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
    NFOLDS = 5
    numiter = 2500

    # DODODODODODO
    train = pd.read_feather('../features/featured/train_full_4.feather')
    y = pd.read_csv('../input/train.csv', usecols=['deal_probability'])
    lentrain = train.shape[0]

    # TFIDF for merge
    with open('./tfidf/kernel_tfidf.pickle', 'rb') as f:
        tfidf_feature = pickle.load(f)
    dummy_tfvocab = ['dummy_'+str(i) for i in range(tfidf_feature.shape[1])]
    predictors = list(train.columns) + list(dummy_tfvocab)

    # concat
    train = hstack([csr_matrix(train), tfidf_feature[:lentrain]])

    lgb_models = []
    best_iters = []
    feature_imp_split = []
    feature_imp_gain = []
    preds = pd.DataFrame()
    print(train.shape)
    print(y.shape)
    trainX, validX, trainY, validY = train_test_split(train , y, test_size=0.2)
    print(trainX.shape, validX.shape, trainY.shape, validY.shape)

    #Dtrain, Dvalid = get_lgb_train(train.iloc[train_idx,:], y.iloc[train_idx,:], train.iloc[valid_idx,:], y.iloc[valid_idx,:], predictors, categorical)
    #Dtrain, Dvalid = get_lgb_train(train[train_idx], y.iloc[train_idx,:], train[valid_idx], y.iloc[valid_idx,:], predictors, categorical)
    #Dtrain, Dvalid = get_lgb_train(trainX, trainY, validX, validY, predictors=predictors, categorical=categorical)
    Dtrain = lgb.Dataset(trainX, label=trainY.values.ravel(), feature_name=predictors, categorical_feature=categorical)
    Dvalid = lgb.Dataset(validX, label=validY.values.ravel(), feature_name=predictors, categorical_feature=categorical)    
    del train, y; gc.collect()
    lgb_model, bestiter = get_lgb_model(Dtrain, Dvalid, categorical=categorical, numiter=numiter, isValid=True, randomseed=76)
    del Dtrain, Dvalid; gc.collect()

    # DODODODO
    if EnableSubmit == True:
        test = pd.read_feather('../features/featured/test_full_4.feather')
        test = hstack([csr_matrix(test), tfidf_feature[lentrain:]])
        preds['lgb_pred_kfold_'+str(0)] = lgb_model.predict(test, num_iteration=bestiter)
        del test; gc.collect()
    
    lgb_models.append(lgb_model)
    best_iters.append(bestiter)
    fcols = lgb_model.feature_name()
    fimps = lgb_model.feature_importance(importance_type='split')
    feature_imp_split.append({fcol: fimp for fimp, fcol in zip(fimps, fcols)})
    fimps = lgb_model.feature_importance(importance_type='gain')
    feature_imp_gain.append({fcol: fimp for fimp, fcol in zip(fimps, fcols)})
    
    # DODDODODOO
    train = pd.read_feather('../features/featured/train_full_4.feather')
    train = hstack([csr_matrix(train), tfidf_feature[:lentrain]])
    y = pd.read_csv('../input/train.csv', usecols=['deal_probability'])

    del train, y
    gc.collect()

    print('Best iterations:')
    print(best_iters)

    '''
    datetime = dt.now().strftime('%Y_%m%d_%H%M_%S')
    fileprefix = '../subs/lightgbm_kfold_ensemble_'

    if EnableSubmit == True:
        subs = pd.read_csv('../input/test.csv', usecols=['item_id'])

        pred_ensemble = np.zeros_like(preds['lgb_pred_kfold_0'].values)
        for col in preds.columns:
            preds[col] = np.clip(preds[col], 0.0000001, 1)
            pred_ensemble += (np.log(preds[col]).values) / np.float(NFOLDS)
        pred_ensemble = np.exp(pred_ensemble)
        pred_ensemble = np.clip(pred_ensemble, 0, 1)
        subs['deal_probability'] = pred_ensemble
        #subs['deal_probability'].fillna(0, inplace=True)

        filename = fileprefix+datetime+'.csv.gz'
        subs.to_csv(filename, index=False, float_format='%.9f', compression='gzip')

        del subs; gc.collect()
    '''
    '''
    # WHOLE
    numiter = np.uint(np.mean(best_iters)* 1.1)
    print('WHOLE TRAIN: ', numiter)

    # DODODDODO
    train = pd.read_feather('../features/featured/train_full.feather')
    y = pd.read_csv('../input/train.csv', usecols=['deal_probability'])
    predictors = train.columns.tolist()
    print(train.shape)
    print(y.shape)

    Dtrain, Dvalid = get_lgb_train(train, y, train.iloc[:2, :], y.iloc[:2, :], predictors, categorical)
    del train, y; gc.collect()
    lgb_model, best_iter = get_lgb_model(Dtrain, Dvalid, numiter=numiter, isValid=False, randomseed=98)
    del Dtrain; gc.collect()

    feature_imp_split = []
    feature_imp_gain = []
    fcols = lgb_model.feature_name()
    fimps = lgb_model.feature_importance(importance_type='split')
    feature_imp_split.append({fcol: fimp for fimp, fcol in zip(fimps, fcols)})
    fimps = lgb_model.feature_importance(importance_type='gain')
    feature_imp_gain.append({fcol: fimp for fimp, fcol in zip(fimps, fcols)})

    datetime = dt.now().strftime('%Y_%m%d_%H%M_%S')
    fileprefix = '../subs/lightgbm_kfold_ensemble_'

    if EnableSubmit == True:
        subs = pd.read_csv('../input/test.csv', usecols=['item_id'])
        # DODODDODODODO
        test = pd.read_feather('../features/featured/test_full.feather')
        subs['deal_probability'] = lgb_model.predict(test[predictors], num_iteration=best_iter)

        filename = fileprefix+datetime+'.csv.gz'
        subs.to_csv(filename, index=False, float_format='%.9f', compression='gzip')
        del subs, test; gc.collect()    
    '''