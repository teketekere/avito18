# resnet50, inceptionv3, xception, vgg19, densenet201 
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121
from keras.applications.mobilenet import MobileNet 
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import numpy.matlib
import pandas as pd
import os
import re
import gc
from collections import Counter
from tqdm import tqdm
from imageFeaturesExtractionMulti import check_imgpath, load_image
from myutils import timer

tqdm.pandas()

def predict_imagenet(img_path, models, topk=3):
    defaultret = np.matlib.repmat([0, 0, 0], topk, 1)
    if check_imgpath(img_path) == False:
        return [defaultret for _ in range(len(models))]
    try:
        img = image.load_img(img_path, target_size=(224, 224))
    except Exception as e:
        print('Cannot open image: ', img_path)
        return [defaultret for _ in range(len(models))]
    try:
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    except Exception as e:
        print('Cannot resize: ', img_path)
        return [defaultret for _ in range(len(models))]

    preds = []
    for model in models:
        pred = model.predict(x)
        pred = decode_predictions(pred, top=topk)[0]
        #pred = [p[2] for p in pred]
        preds.append(pred)
    return preds

if __name__ == '__main__':
    topk = 3
    model_dict = {#'Resnet50': ResNet50,
                  #'IndeptionV3': InceptionV3,
                  #'Xception': Xception,
                  #'VGG19': VGG19,
                  #'VGG16': VGG16,
                  'Movilenet': MobileNet,
                  #'DenseNet121': DenseNet121
                  }

    features_path = '../features/'
    input_path = '../input/'
    train = pd.read_csv('../input/train.csv', usecols=['image'])
    test = pd.read_csv('../input/test.csv', usecols=['image'])

    #train = pd.read_csv('../input/train.csv', usecols=['image'], nrows=5)
    #test = pd.read_csv('../input/test.csv', usecols=['image'], nrows=7)

    print(train.isnull().sum())
    print(test.isnull().sum())
    lentrain = train.shape[0]
    train['image'] = train['image'].fillna('')
    test['image'] = test['image'].fillna('')
    train['image'] = train['image'].apply(lambda x: input_path+'train_jpg/'+str(x)+'.jpg')
    test['image'] = test['image'].apply(lambda x: input_path+'test_jpg/'+str(x)+'.jpg')

    features = pd.concat([train, test])
    features.reset_index(drop=True, inplace=True)
    print(features.shape)

    imagenet_class = pd.DataFrame()
    for key, model in model_dict.items():
        models = []
        models.append(model(weights='imagenet'))
        with timer('predicting '+key):
            features['preds_' + key] = features['image'].progress_apply(lambda x: predict_imagenet(x, models, topk=topk))

        imagenet_preds = features['preds_'+key].values.ravel()
        temp_df = pd.DataFrame()
        for t in range(topk):
            temp_df['imagenet_'+key+'_top'+str(t+1)] = [val[0][t][2] for val in imagenet_preds]
        train_df = temp_df[:lentrain]
        test_df = temp_df[lentrain:]
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)
        print(train_df.shape)
        print(test_df.shape)
        train_df.to_feather('../features/train/imagenet_'+key+'_train_debug.feather')
        test_df.to_feather('../features/test/imagenet_'+key+'_test_debug.feather')

        # 2個めのインデックスをk-1に変えればtopk
        imagenet_class['imagenet_'+key+'_class1'] = [val[0][0][1] for val in imagenet_preds]

        del train_df, test_df, temp_df
        #del test_df, temp_df
        gc.collect()
        print('Save ', key)

        del models; gc.collect()

    '''
    print(imagenet_class.shape)
    num_models = np.float(len(model_dict))
    top1_confidence_rate = []
    for i in range(imagenet_class.shape[0]):
        pred_class = [imagenet_class['imagenet_'+key+'_class1'][i] for key in model_dict.keys()]
        pred_class_dict = Counter(pred_class)
        pred_class_dict_sorted = sorted(pred_class_dict.items(), key=lambda x: -x[1])
        pred_confidence_rate = pred_class_dict_sorted[0][1] / num_models if pred_class_dict_sorted[0][0] != 0 else 0.0
        top1_confidence_rate.append(pred_confidence_rate)


    top1_confidence = pd.DataFrame()
    top1_confidence['imagenet_top1_rate'] = top1_confidence_rate
    print(top1_confidence.shape)
    train_df = top1_confidence[:lentrain]
    test_df = top1_confidence[lentrain:]
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    print(train_df.shape)
    print(test_df.shape)
    train_df.to_feather('../features/train/imagenet_top1_rate_train_debug.feather')
    test_df.to_feather('../features/test/imagenet_top1_rate_test_debug.feather')
    '''
    '''
    for key in model_dict.keys():
        imagenet_preds = features['preds_'+key].values.ravel()
        for t in range(topk):
            features['imagenet_'+key+'_top'+str(t+1)] = [val[0][t] for val in imagenet_preds]

    print(features.shape)

    predcols = [col for col in features.columns if str(col).split('_')[0] == 'preds']
    print(predcols)
    features = features.drop(predcols+['image'], axis=1)
    train = features[: lentrain]
    test = features[lentrain:]

    print(train.shape)
    print(test.shape)

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    train.to_feather('../features/train/imagenet_train.feather')
    test.to_feather('../features/test/imagenet_test.feather')
    '''