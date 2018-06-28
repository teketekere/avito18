from collections import defaultdict
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image as IMG
import numpy as np
import pandas as pd 
import operator
import cv2
import os 
import gc
from multiprocessing import Pool
import multiprocessing
from copy import deepcopy
import time
from contextlib import contextmanager
import pickle
from myutils import timer, reduce_mem_usage

# global vars
## detect this path depends on envs
#images_path = '../input/ants/'
#images_path = '../input/hoge/'

features_path = '../features/'
input_path = '../input/'

# define functions
def check_imgpath(img):
    return os.path.isfile(img)

def load_image(img, usecv2=False):
    path = img
    try:
        im = cv2.imread(path) if usecv2==True else IMG.open(path)
    except Exception as e:
        print('Cannot open img: ', img)    
    return im

def crop_horizontal(im):
    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0]/2, size[1]/2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))
    return im1, im2

def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1

    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent

def perform_color_analysis_black(img):
    if check_imgpath(img) ==False:
        return -1

    im = load_image(img)
    im1, im2 = crop_horizontal(im)

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        print('Calculation error')
        return None

    dark_percent = (dark_percent1 + dark_percent2)/2 
    return dark_percent

def perform_color_analysis_white(img):
    if check_imgpath(img) ==False:
        return -1

    im = load_image(img)
    im1, im2 = crop_horizontal(im)

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        print('Calculation error')
        return None

    light_percent = (light_percent1 + light_percent2)/2 
    return light_percent

def average_pixel_width(img):
    if check_imgpath(img) ==False:
        return -1

    im = load_image(img)    
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100

def get_dominant_color(img):
    if check_imgpath(img) ==False:
        return [-1, -1, -1]

    im = load_image(img, usecv2=True)
    arr = np.float32(im)
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(im.shape)

    dominant_color = palette[np.argmax(np.unique(labels, return_counts=True)[1: ])]
    return dominant_color

def get_average_color(img):
    if check_imgpath(img) ==False:
        return [-1, -1, -1]

    im = load_image(img, usecv2=True)
    average_color = [im[:, :, i].mean() for i in range(im.shape[-1])]
    return average_color

def get_size(img):
    if check_imgpath(img) ==False:
        return -1

    filename = img
    st = os.stat(filename)
    return st.st_size

def get_dimensions(img):
    if check_imgpath(img) ==False:
        return [-1, -1]

    im = load_image(img)
    img_size = im.size
    return img_size

def get_blurrness_score(img):
    if check_imgpath(img) ==False:
        return -1

    im = load_image(img, usecv2=True)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(im, cv2.CV_64F).var()
    return fm

def get_arraylike_features(img):
    methods = [perform_color_analysis_black,
               perform_color_analysis_white,
               average_pixel_width,
               #get_dominant_color,
               get_average_color,
               get_size,
               get_dimensions,
               get_blurrness_score
              ]
    results = []
    for m in methods:
        results.append(m(img))
    return results

def get_imagefeatures_multi(features, imgcol, prefix='', n_workers=1):
    # todo: get_arraylike_featuresの引数にmethodsを渡してpartialで固定する
    prefix = prefix + '_'
    print(features.shape[0])
    print(features.isnull().sum())
    #params = [features[imgcol][i] for i in range(features.shape[0])]
    params = features[imgcol].tolist()
    print(len(params))
    with timer('Image features extraction'):
        with Pool(processes=n_workers) as pool:
            results = pool.map(get_arraylike_features, params)

    # transform into pd.df
    features[prefix+'dullness'] = [results[i][0] for i in range(features.shape[0])]
    features[prefix+'whiteness'] = [results[i][1] for i in range(features.shape[0])]
    features[prefix+'average_pixel_width'] = [results[i][2] for i in range(features.shape[0])]
    #features[prefix+'dominant_color'] = [results[i][3] for i in range(features.shape[0])]
    features[prefix+'average_color'] = [results[i][3] for i in range(features.shape[0])]
    features[prefix+'image_size'] = [results[i][4] for i in range(features.shape[0])]
    features[prefix+'dim_size'] = [results[i][5] for i in range(features.shape[0])]
    features[prefix+'blurrness'] = [results[i][6] for i in range(features.shape[0])]

    
    # adhook
    #features[prefix+'dominant_red'] = features[prefix+'dominant_color'].apply(lambda x: x[0]) / 255
    #features[prefix+'dominant_green'] = features[prefix+'dominant_color'].apply(lambda x: x[1]) / 255
    #features[prefix+'dominant_blue'] = features[prefix+'dominant_color'].apply(lambda x: x[2]) / 255
    features[prefix+'average_red'] = features[prefix+'average_color'].apply(lambda x: x[0]) / 255
    features[prefix+'average_green'] = features[prefix+'average_color'].apply(lambda x: x[1]) / 255
    features[prefix+'average_blue'] = features[prefix+'average_color'].apply(lambda x: x[2]) / 255
    features[prefix+'width'] = features[prefix+'dim_size'].apply(lambda x : x[0])
    features[prefix+'height'] = features[prefix+'dim_size'].apply(lambda x : x[1])
    #features.drop(prefix+'dominant_color', axis=1, inplace=True)        
    features.drop(prefix+'average_color', axis=1, inplace=True)
    features.drop(prefix+'dim_size', axis=1, inplace=True)
    gc.collect()
    return features


def save_features(features, filename):
    filepath = features_path + filename
    features.to_feather(filepath)

if __name__ == '__main__':
    #imgs = os.listdir(images_path)
    #features = pd.DataFrame()
    #features['imagepath'] = imgs
    train = pd.read_csv('../input/train.csv', usecols=['image'])
    test = pd.read_csv('../input/test.csv', usecols=['image'])
    lentrain = train.shape[0]

    train['image'] = train['image'].fillna('')
    test['image'] = test['image'].fillna('')
    train['image'] = train['image'].apply(lambda x: input_path+'train_jpg/'+str(x)+'.jpg')
    test['image'] = test['image'].apply(lambda x: input_path+'test_jpg/'+str(x)+'.jpg')

    #features = train.iloc[:100,:]
    features = pd.concat([train, test])
    print(features.shape)
    del train, test; gc.collect()

    numcpu = multiprocessing.cpu_count()
    print(f'use {numcpu} cpus')
    get_imagefeatures_multi(features, 'image', prefix='imagefeatures_', n_workers=numcpu)

    features.drop('image', axis=1, inplace=True)
    features = reduce_mem_usage(features)
    save_features(features, 'imagefeature.feather')
    print('done')
