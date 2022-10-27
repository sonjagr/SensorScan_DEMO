# Created by Sonja Grönroos, August 2022
#
# Copyright (c) 2021 CERN
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import math
import cv2

from config import *
from sklearn.model_selection import train_test_split

def db_to_path(db):
    db = db.reset_index()
    db['Path'] = db.Campaign+ "/"+ db.DUT+ "/"+db.FileName
    return db['Path'].to_numpy()

def open_db(name):
    f = os.path.join(TRAIN_DIR_LOC, name)  # newest database
    with pd.HDFStore(f, mode='r') as store:
        db = store.select('db')
        #print(f'Reading {f}')
    return db

def save_db(db, name):
    f = os.path.join(TRAIN_DIR_LOC, name)  # newest database
    db.to_hdf(f, key='db', mode = 'w')

def process_anomalous_df_to_numpy(db):
    db = db[db['orig_boxY'].map(len) > 0]
    db = db.reset_index()[['Campaign', 'DUT', 'FileName', 'orig_boxY', 'orig_boxX']]
    db = db.drop(db.loc[(db.Campaign == 'September2021_PM8') & (db.DUT == '8inch_198ch_N3311_7')].index)
    X_list = db_to_path(db)
    db = db.drop(['Campaign', 'DUT', 'FileName'], axis =1)
    db['crop_lbls'] = pd.NaT
    db['crop_lbls'] = db.apply(lambda x: box_to_labels(x.orig_boxX, x.orig_boxY).flatten(), axis=1)
    Y_list = db['crop_lbls'].to_numpy().flatten()
    return X_list, Y_list

## resize images to the required size, crop from bottom
def resize(item):
    model = tf.keras.Sequential([tf.keras.layers.Cropping2D(cropping=((0, 16),(0, 0)))])
    return model(item)

## read npy file for dataset
def read_npy_file(item):
    data = np.load(item.numpy().decode())
    data = np.expand_dims(data, axis=2)
    return data.astype(np.float32)

def tf_read_npy_file(item):
  im_shape = item.shape
  [item] = tf.py_function(read_npy_file, [item], [tf.float32])
  item.set_shape(im_shape)
  return [item]

## create a general dataset
def create_dataset(file_list, _shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices(file_list)
    if _shuffle:
        dataset.shuffle(len(file_list), reshuffle_each_iteration=True)
    return dataset.map(lambda item: tuple(tf.py_function(read_npy_file, [item], [tf.float32,])))

## creates a training dataset for the CNN
def create_cnn_dataset(file_list, label_list, _shuffle=False):
    imgs = tf.data.Dataset.from_tensor_slices(file_list)
    lbls = tf.data.Dataset.from_tensor_slices(label_list)
    dataset = tf.data.Dataset.zip((imgs, lbls))
    if _shuffle:
        dataset.shuffle(len(file_list), seed = 42, reshuffle_each_iteration=True)
    images = dataset.map(lambda x, y: x)
    labels = dataset.map(lambda x, y:  tf.cast(y, tf.float32))
    images = images.map(lambda item: tuple(tf.py_function(read_npy_file, [item], [tf.float32,])))
    dataset = tf.data.Dataset.zip((images, labels))
    return dataset

def format_new_anomalous_dataset(new_db, date_of_previous_train):
    new_db = new_db.reset_index()
    new_db['Step'] = new_db['FileName'].str.split('step', 1, expand=True)[1]
    new_db['FileName'] = new_db['FileName'].astype(str) + '.npy'
    new_db = new_db.groupby(['Campaign', 'DUT','FileName', 'ValDate', 'Step'])[['x', 'y']].agg(lambda x: list(x))
    new_db = new_db.reset_index().set_index(['Campaign', 'DUT', 'Step'])
    new_db = new_db.rename(columns={"x": "orig_boxX", "y": "orig_boxY", 'ValDate': 'Date'})

    new_date = new_db["Date"].max()
    mask = (new_db['Date'] > date_of_previous_train)
    new_db = new_db.loc[mask]
    print(new_db)

    new_anom_train_db, new_test_val_db = train_test_split(new_db, test_size=0.2, shuffle = True, random_state = 42)
    new_anom_val_db, new_anom_test_db = train_test_split(new_test_val_db, test_size=0.5, shuffle = True, random_state = 42)

    return new_anom_train_db, new_anom_val_db, new_anom_test_db, new_date, new_db.reset_index()['FileName'].tolist()

def format_normal_dataset(norm_ds, date_of_previous_train):
    mask = (norm_ds['ValDate'] > date_of_previous_train)
    norm_ds = norm_ds.loc[mask]
    print(norm_ds)

    new_norm_train_db, new_test_val_db = train_test_split(norm_ds, test_size=0.2, shuffle=True, random_state=42)
    new_norm_val_db, new_norm_test_db = train_test_split(new_test_val_db, test_size=0.5, shuffle=True, random_state=42)
    return new_norm_train_db, new_norm_val_db, new_norm_test_db

def combine_normal_datasets(new_norm_db, name):
    new_date = new_norm_db["ValDate"].max()
    old_norm_npy = np.load(os.path.join("db\DET", "NORMAL_%s_20220711.npy" % name), allow_pickle=True)
    new_norm_npy = db_to_path(new_norm_db)
    combined_norm_npy = np.unique(np.append(old_norm_npy, new_norm_npy))
    np.save(os.path.join("db\DET", "NORMAL_%s_%s.npy" % (name, str(new_date))), combined_norm_npy)
    print('%s new normal images have been added to normal image %s database' % (len(new_norm_npy), name))

def combine_datasets(old_anom_db, new_anom_db, name):
    old_anom_db = old_anom_db[['FileName', 'Date', 'orig_boxX', 'orig_boxY']]
    combined_db = pd.concat([old_anom_db, new_anom_db])
    new_date = combined_db["Date"].max()
    init_len = len(old_anom_db.Date.tolist())
    combined_db = combined_db[~combined_db.index.duplicated(keep='last')]
    print(combined_db)
    fin_len = len(combined_db.Date.tolist())
    save_db(combined_db, "%s_DATABASE" % name)
    print('%s new anomalous images have been added to %s_DATABASE, newest data was collected on %s' % (fin_len-init_len, name, str(new_date)))

    return combined_db

## convert box coordinates to labels
def box_to_labels(BoxX, BoxY):
    labels = np.zeros((REDUCED_DIMENSION[0], REDUCED_DIMENSION[1]))
    for i, x in enumerate(BoxX):
        x = int(x)
        y = int(BoxY[i])
        if (np.mod(x, PATCHSIZE) == 0) and (np.mod(y, PATCHSIZE) == 0):
            xi = int(x / PATCHSIZE)
            yi = int(y / PATCHSIZE)
            labels[yi, xi] = 1
        else:
            print('Error in annotation conversion')
    return labels

## convert box index to the coordinates
def box_index_to_coords(box_index):
    row = math.floor(box_index / REDUCED_DIMENSION[1])
    col = box_index % REDUCED_DIMENSION[1]
    return col * PATCHSIZE, row * PATCHSIZE

## convert rgb to bayer format
def rgb2bayer(rgb):
    (h,w) = rgb.shape[0], rgb.shape[1]
    (r,g,b) = cv2.split(rgb)
    bayer = np.empty((h,w), np.uint8)
    bayer[0::2, 0::2] = r[0::2, 0::2]
    bayer[0::2, 1::2] = g[0::2, 1::2]
    bayer[1::2, 0::2] = g[1::2, 0::2]
    bayer[1::2, 1::2] = b[1::2, 1::2]
    return bayer





