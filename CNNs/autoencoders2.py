# Created by Thorben Quast, August 2021
# Modified by Sonja Grönroos, August 2022
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

import os
import tensorflow as tf
from config import *

# define the encoder and decoder networks
class AutoEncoder(tf.keras.Model):
    """Convolutional DNN-based autoencoder."""

    def __init__(self):
        super(AutoEncoder, self).__init__()

    def initialize_network_TQ3_2(self):
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(
                    2720, PICTURESIZE_X, 1)),
                tf.keras.layers.experimental.preprocessing.Rescaling(1. / EIGHTBITMAX),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=(5, 4), strides=(5, 4), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=(2, 2), strides=(2, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(4, 2), strides=(4, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(2, 5), strides=(2, 5), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(2, 2), strides=(2, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(1, 1), strides=(1, 1), activation='elu', padding="valid", use_bias=True),
            ]
        )
        print(self.encoder.summary())

        self.decoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=self.encoder.output.shape[1:]),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(1, 1), strides=(1, 1), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(2, 2), strides=(2, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(2, 5), strides=(2, 5), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=(4, 2), strides=(4, 2), activation='elu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=(2, 2), strides=(2, 2), activation='relu', padding="valid", use_bias=True),
                tf.keras.layers.Conv2DTranspose(
                    filters=1, kernel_size=(5, 4), strides=(5, 4), activation='relu', padding="valid", use_bias=True),
                tf.keras.layers.experimental.preprocessing.Rescaling(EIGHTBITMAX),
            ]
        )
        print(self.decoder.summary())

    def encode(self, im):
        return self.encoder(im)

    def decode(self, z):
        return self.decoder(z)

    def save(self, path):
        encoder_path = path + "_encoder"
        self.encoder.save(encoder_path)

        decoder_path = path + "_decoder"
        self.decoder.save(decoder_path)

    def load(self, fpath):
        encoder_path = fpath + "_encoder"
        if os.path.exists(encoder_path):
            self.encoder = tf.keras.models.load_model(encoder_path, compile=True)
        else:
            return False

        decoder_path = fpath + "_decoder"
        self.decoder = tf.keras.models.load_model(decoder_path, compile=True)
        print("Loaded network from", fpath)
        return True
