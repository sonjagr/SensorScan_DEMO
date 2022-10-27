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

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import *

def plot_metrics(history, savename):
    colors = ['blue','red']
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric], color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.grid()
        plt.legend()
    plt.tight_layout()
    plt.savefig('saved_CNNs/%s/training.png' % savename, dpi = DPI)
    plt.show()

def plot_examples(ds, label = False):
    for x, y in ds:
        dim = tf.shape(x)[-1]
        x = tf.reshape(x, [PATCHSIZE, PATCHSIZE, dim])
        x = x.numpy()
        plt.imshow(x.astype('uint8'), cmap = 'gray_r',vmin = 0, vmax = 250.)
        if label:
            plt.title(str(y))
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False,left=False, labelleft=False)
        plt.tight_layout()
        plt.show()

@tf.function
def crop(img, lbl):
    img = tf.reshape(img, [-1, PICTURESIZE_Y+16, PICTURESIZE_X, 1])
    img = tf.keras.layers.Cropping2D(cropping=((0, 16), (0, 0)))(img)
    return img, lbl

@tf.function
def flip(image_label, seed):
    image, label = image_label
    INPUT_DIM = tf.shape(image)[-1]
    image = tf.reshape(image, [PATCHSIZE, PATCHSIZE, INPUT_DIM])
    if seed > 5:
        flipped = tf.image.flip_left_right(image)
    else:
        flipped = tf.image.flip_up_down(image)
    return tf.reshape(flipped, [PATCHSIZE, PATCHSIZE, INPUT_DIM]), label

@tf.function
def flip_h(image, label):
    INPUT_DIM = tf.shape(image)[-1]
    image = tf.reshape(image, [PATCHSIZE, PATCHSIZE, INPUT_DIM])
    flipped = tf.image.flip_left_right(image)
    return tf.reshape(flipped, [PATCHSIZE, PATCHSIZE, INPUT_DIM]), label

@tf.function
def flip_v(image, label):
    INPUT_DIM = tf.shape(image)[-1]
    image = tf.reshape(image, [PATCHSIZE, PATCHSIZE, INPUT_DIM])
    flipped = tf.image.flip_up_down(image)
    return tf.reshape(flipped, [PATCHSIZE, PATCHSIZE, INPUT_DIM]), label

@tf.function
def patch_images(img, lbl):
    INPUT_DIM = tf.shape(img)[-1]
    split_img = tf.image.extract_patches(images=img, sizes=[1, PATCHSIZE, PATCHSIZE, 1], strides=[1, PATCHSIZE, PATCHSIZE, 1], rates=[1, 1, 1, 1], padding='VALID')
    re = tf.reshape(split_img, [PATCHES, PATCHSIZE * PATCHSIZE, INPUT_DIM])
    lbl = tf.reshape(lbl, [PATCHES])
    patch_ds = tf.data.Dataset.from_tensors((re, lbl))
    return patch_ds

@tf.function
def patch_image(img):
    split_img = tf.image.extract_patches(images=img, sizes=[1, PATCHSIZE, PATCHSIZE, 1], strides=[1, PATCHSIZE, PATCHSIZE, 1],
                                         rates=[1, 1, 1, 1], padding='VALID')
    re = tf.reshape(split_img, [17 * 24, PATCHSIZE * PATCHSIZE])
    return re

## calculate dataset length
def ds_length(ds):
    ds = ds.as_numpy_iterator()
    ds = list(ds)
    dataset_len = len(ds)
    return dataset_len

## convert rgb to bayer format
def rgb2bayer(rgb):
    rgb= rgb.numpy()
    (h,w) = rgb.shape[0], rgb.shape[1]
    (r,g,b) = cv2.split(rgb)
    bayer = np.empty((h, w), np.uint8)
    bayer[0::2, 0::2] = r[0::2, 0::2]
    bayer[0::2, 1::2] = g[0::2, 1::2]
    bayer[1::2, 0::2] = g[1::2, 0::2]
    bayer[1::2, 1::2] = b[1::2, 1::2]
    return bayer

## a wrapper for rgb2bayer
def tf_rgb2bayer(image):
  im_shape = image.shape
  [image] = tf.py_function(rgb2bayer, [image], [tf.uint8])
  image.set_shape(im_shape)
  return tf.reshape(image, [-1])

## convert bayer to rgb
def bayer2rgb(bayer):
    bayer = bayer.numpy()
    shape = np.shape(bayer)
    rgb = cv2.cvtColor(bayer.reshape(shape[1],shape[2],1).astype('uint8'), cv2.COLOR_BAYER_RG2RGB)
    return rgb

## a wrapper for bayer2rgb
def tf_bayer2rgb(bayer):
    rgb = tf.py_function(bayer2rgb, [bayer], [tf.float32])
    return rgb

@tf.function
def encode(img, lbl, ae):
    INPUT_DIM = tf.shape(img)[-1]
    img = tf.reshape(img, [-1, PICTURESIZE_Y, PICTURESIZE_X, INPUT_DIM])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = tf.abs(tf.subtract(img, decoded_img))
    return aed_img, lbl

@tf.function
def bright_encode(img, lbl, ae, delta):
    INPUT_DIM = tf.shape(img)[-1]
    img = tf.cast(img, tf.float64)
    img = tf.math.multiply(img, delta)
    img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=EIGHTBITMAX)
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, [-1, PICTURESIZE_Y, PICTURESIZE_X, INPUT_DIM])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    aed_img = tf.abs(tf.subtract(img, decoded_img))
    return aed_img, lbl

@tf.function
def bright(img, lbl, delta):
    INPUT_DIM = tf.shape(img)[-1]
    img = tf.cast(img, tf.float64)
    img = tf.math.multiply(img, delta)
    img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=EIGHTBITMAX)
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, [-1, PICTURESIZE_Y, PICTURESIZE_X, INPUT_DIM])
    return img, lbl

@tf.function
def encode_rgb(img, lbl, ae):
    img = tf.reshape(img, [-1, PICTURESIZE_Y, PICTURESIZE_X, 1])
    img_rgb = tf_bayer2rgb(img)
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    decoded_img_rgb = tf_bayer2rgb(decoded_img)
    aed_img = tf.abs(tf.subtract(img_rgb, decoded_img_rgb))
    return aed_img, lbl

@tf.function
def bright_encode_rgb(img, lbl, ae, delta):
    img = tf.cast(img, tf.float64)
    img = tf.math.multiply(img, delta)
    img = tf.clip_by_value(img, clip_value_min=0, clip_value_max=EIGHTBITMAX)
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, [-1, PICTURESIZE_Y, PICTURESIZE_X, 1])
    encoded_img = ae.encode(img)
    decoded_img = ae.decode(encoded_img)
    img_rgb = tf_bayer2rgb(img)
    decoded_img_rgb = tf_bayer2rgb(decoded_img)
    aed_img = tf.abs(tf.subtract(img_rgb, decoded_img_rgb))
    return aed_img, lbl

@tf.function
def rotate(image_label, rots):
    image, label = image_label
    INPUT_DIM = tf.shape(image)[-1]
    image = tf.reshape(image, [PATCHSIZE, PATCHSIZE, INPUT_DIM])
    rot = tf.image.rot90(image, k=rots)
    return tf.reshape(rot, [PATCHSIZE, PATCHSIZE, INPUT_DIM]), label

@tf.function
def crop_resize(image_label, seed):
    image, label = image_label
    INPUT_DIM = tf.shape(image)[-1]
    image = tf.image.random_crop(value=image, size = (PATCHSIZE - 20, PATCHSIZE - 20, INPUT_DIM), seed =seed)
    image = tf.image.resize(image, [PATCHSIZE, PATCHSIZE, INPUT_DIM])
    return image, label

@tf.function
def format_data(image, label):
    INPUT_DIM = tf.shape(image)[-1]
    image = tf.reshape(image, [PATCHSIZE, PATCHSIZE, INPUT_DIM])
    label = tf.cast(label, tf.float32)
    return image, label

@tf.function
def format_data_batch(image, label):
    image = tf.reshape(image, [PATCHES, PATCHSIZE, PATCHSIZE, 1])
    label = tf.cast(label, tf.float32)
    return image, label