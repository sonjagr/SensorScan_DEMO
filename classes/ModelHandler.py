# Created by Sonja Grönroos, August 2022
#
# Copyright (c) 2021 CERN
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import json
import numpy as np
import pandas as pd
import neptune.new as neptune
import random
import time
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import sklearn
from tensorflow import keras
import seaborn as sns
import pickle

from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from CNNs.autoencoders2 import *
from sklearn.metrics import confusion_matrix
from utility.cnn_helpers import (crop, encode, bright, bright_encode, flip, format_data,
                                 format_data_batch, patch_image, patch_images,
                                 rotate)
from utility.fs import create_dir
from utility.keras import setTrainable
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from config import *
from utility.dataset_helpers import (box_index_to_coords, create_cnn_dataset,
                                     process_anomalous_df_to_numpy)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

METRICS = [
              tf.keras.metrics.TruePositives(name='tp'),
              tf.keras.metrics.FalsePositives(name='fp'),
              tf.keras.metrics.TrueNegatives(name='tn'),
              tf.keras.metrics.FalseNegatives(name='fn'),
              tf.keras.metrics.Precision(name='precision'),
              tf.keras.metrics.Recall(name='recall'),
              tf.keras.metrics.AUC(name='auc'),
              tf.keras.metrics.AUC(name='prc', curve='PR'),
            ]
random.seed(42)

class ModelSavingCallback(keras.callbacks.Callback):
    def __init__(self, filepath):
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        self.trainParam_dictionary['current_epoch'] = epoch
        self.saveTrainParameters()

class NeptuneMonitoringCallback(keras.callbacks.Callback):
    def on_epoch_end(self, logs={}):
        for metric_name, metric_value in logs.items():
            neptune.log_metric(metric_name, metric_value)

class Modelhandler:
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.br_model = None
        self.run = None
        self.images_dir_loc = IMAGES_DIR_LOC
        self.train_db_loc = TRAIN_DIR_LOC
        self.ae_location = os.path.join(MODELS_DIR_LOC, "AE3_v2_277_epochs")
        self.ae = None
        self.newest_val_date = None
        self.verbose = True
        self.br_th = 0.5
        self.mode = None
        self.trainParam_dictionary = {}
        self.testing_save_loc = None

    def loadClassifierModel(self, model_name, filepath, summary=False):
        self.model_saving_filepath = filepath
        self.saving_path = os.path.join(MODELS_DIR_LOC, "class_cnn_test_outputs", model_name)
        create_dir(self.saving_path)
        self.model_name = model_name
        self.classifier = keras.models.load_model(os.path.join(MODELS_DIR_LOC, model_name) )
        if summary:
            print(self.classifier.summary())
        setTrainable(self.classifier, False)
        return self.classifier

    def loadBRModel(self, model_loc, summary=False):
        self.br_model = keras.models.load_model(os.path.join(model_loc))
        if summary:
            print(self.br_model.summary())
        setTrainable(self.br_model, False)
        return self.br_model

    def loadTrainParameters_json(self, filename):
        try:
            with open(os.path.join(self.saving_path, filename), 'r') as file:
                self.trainParam_dictionary = json.load(file)
                print("Loaded train parameters: ", self.trainParam_dictionary)
        except:
            print('ERROR: Train parameter file not read')

    def saveTrainParameters(self, filename):
        with open(os.path.join(self.saving_path, filename), 'w') as file:
            file.write(json.dumps(self.trainParam_dictionary))

    def initTrainParameters(self, th, bs, lr, optimizerName, lossFunc, normal_to_anom, epochs, cont_epoch, fl_gamma, bright_aug):
        self.trainParam_dictionary = {
            "threshold": th,
            "batch_size": bs,
            "lr": lr,
            "optimizer" : optimizerName,
            "lossFunc": lossFunc,
            "normal_to_anom" : normal_to_anom,
            "epochs" : epochs,
            "fl_gamma": fl_gamma,
            "bright_aug": bright_aug,
            "current_epoch" : cont_epoch
        }
        with open(os.path.join(self.saving_path, 'trainParams_original.pkl'), 'w') as file:
            file.write(json.dumps(self.trainParam_dictionary))

    def reinit_model(self):
        self.classifier = tf.keras.models.clone_model(self.classifier)

    def set_mode(self, mode):
        self.mode = mode

    def set_classification_th(self, th):
        self.th = th

    def create_testing_save_loc(self, path):
        create_dir(path)
        self.testing_save_loc = path

    @tf.function
    def process_crop_encode(self, image, label):
        #image, label = crop(image, label)
        image, label = encode(image, label, self.ae)
        return image, label

    @tf.function
    def process_crop_bright_encode(self, image_label, delta):
        image, label = image_label
        image, label = crop(image, label)
        image, label = bright_encode(image, label, self.ae, delta)
        return image, label

    @tf.function
    def process_crop(self, image, label):
        image, label = crop(image, label)
        return image, label

    @tf.function
    def process_crop_bright(self, image_label, delta):
        image, label = image_label
        image, label = crop(image, label)
        image, label = bright(image, label, delta)
        return image, label

    def filter_anom(self, x):
        return tf.math.equal(x, 1)

    def filter_norm(self, x):
        return tf.math.equal(x, 0)

    def init_training(self, filepath, train_db, val_db):
        tf.keras.backend.clear_session()
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)

        self.run = neptune.init(
            tags=['training'],
            with_id='HEX-42',
            project="sgroenro/HexaScanWithAI",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyOGNiZjUwYy1lZTE4LTQ0YWItODEwMy01ZDYxYmZjM2VjYWUifQ==",
        )

        train_hyperparameters = self.trainParam_dictionary
        self.run['train_hyperparameters'] = train_hyperparameters
        np.random.seed(42)

        self.ae = AutoEncoder()
        self.ae.load(self.ae_location)

        ## process data into numpy arrays
        X_train_det_list, Y_train_det_list = process_anomalous_df_to_numpy(train_db)
        X_val_det_list, Y_val_det_list = process_anomalous_df_to_numpy(val_db)

        ## load normal images
        X_train_normal_list = np.load(os.path.join(self.train_db_loc, "NORMAL_TRAIN_20220711.npy"), allow_pickle=True)
        X_val_normal_list = np.load(os.path.join(self.train_db_loc, "NORMAL_VAL_20220711.npy"), allow_pickle=True)

        if self.verbose:
            print()
            print("--- TRAINING: Loading data ---")
            train = len(X_train_normal_list)+len(X_val_normal_list)
            print("Available number of normal images for training and validation: %s" % train)

        ## add image location
        X_train_det_list = [os.path.join(self.images_dir_loc, s) for s in X_train_det_list]
        X_val_det_list = [os.path.join(self.images_dir_loc, s) for s in X_val_det_list]

        N_det_val = len(X_val_det_list)
        N_det_train = len(X_train_det_list)

        X_train_normal_list = [os.path.join(self.images_dir_loc, s) for s in X_train_normal_list]
        X_val_normal_list = [os.path.join(self.images_dir_loc, s) for s in X_val_normal_list]

        N_normal_train = len(X_train_normal_list)
        N_normal_val = len(X_val_normal_list)

        if self.verbose:
            print("    Loaded number of anomalous train images: %s, validation images: %s" % (N_det_train, N_det_val))
        if self.verbose:
            print("    Loaded number of normal train images: %s, validation images: %s" % (N_normal_train, N_normal_val))
        time1 = time.time()

        ## only images with defects
        train_ds = create_cnn_dataset(X_train_det_list, Y_train_det_list.tolist(), _shuffle=False)
        val_ds = create_cnn_dataset(X_val_det_list, Y_val_det_list.tolist(), _shuffle=False)

        ## only normal images
        normal_train_ds = create_cnn_dataset(X_train_normal_list, np.full((int(N_normal_train), PATCHES), 0.0))
        normal_val_ds = create_cnn_dataset(X_val_normal_list, np.full((int(N_normal_val), PATCHES), 0.0))

        lr = self.trainParam_dictionary['lr']
        if self.trainParam_dictionary['optimizer'].casefold() == 'adam'.casefold():
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        elif self.trainParam_dictionary['optimizer'].casefold() == 'nadam'.casefold():
            optimizer = tf.keras.optimizers.Nadam(learning_rate=lr)
        else:
            print("Optimizer error: Check train parameters")

        if self.trainParam_dictionary['lossFunc'] == 'bce':
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        elif self.trainParam_dictionary['lossFunc'] == 'fl':
            loss = SigmoidFocalCrossEntropy(gamma=self.trainParam_dictionary['fl_gamma'], alpha=0.25)
        else:
            print("Loss function error: Check train parameters")

        self.classifier.compile(optimizer=optimizer, loss=loss, metrics=METRICS)

        if self.verbose:
            print("All loaded.")
            print()
            print("--- TRAINING: Data processing ---")

        train_ds_orig = train_ds

        if self.verbose:
            print("    Applying autoencoding")

        train_ds = train_ds.map(self.process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds = val_ds.map(self.process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        normal_train_ds = normal_train_ds.map(self.process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        normal_val_ds = normal_val_ds.map(self.process_crop_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.trainParam_dictionary['bright_aug'] == 1:
            if self.verbose: print("    Augmenting brightness of images")
            brightnesses = np.random.uniform(low=0.75, high=1.25, size=np.array(Y_train_det_list).shape[0])
            counter1 = tf.data.Dataset.from_tensor_slices(brightnesses)
            train_ds_to_bright = tf.data.Dataset.zip((train_ds_orig, counter1))
            train_ds_brightness = train_ds_to_bright.map(lambda x, z: self.process_crop_bright_encode(x, z), num_parallel_calls=tf.data.experimental.AUTOTUNE,)
            train_ds_brightness = train_ds_brightness.flat_map(patch_images).unbatch()

        if self.verbose: print("    Patching")
        train_ds = train_ds.flat_map(patch_images).unbatch()
        val_ds = val_ds.flat_map(patch_images).unbatch()
        normal_train_ds = normal_train_ds.flat_map(patch_images).unbatch()
        normal_val_ds = normal_val_ds.flat_map(patch_images).unbatch()

        train_ds_anomaly = train_ds.filter(lambda x, y: self.filter_anom(y))
        val_ds_anomaly = val_ds.filter(lambda x, y: self.filter_anom(y))

        nbr_anom_train_patches = len(list(train_ds_anomaly))
        nbr_anom_val_patches = len(list(val_ds_anomaly))

        if self.trainParam_dictionary['bright_aug'] == 1:
            train_ds_brightness_anomaly = train_ds_brightness.filter(lambda x, y: self.filter_anom(y))
            train_ds_anomaly = train_ds_anomaly.concatenate(train_ds_brightness_anomaly)

        if self.verbose:
            print("    Number of anomalous training, validation patches: ", nbr_anom_train_patches,nbr_anom_val_patches,)

        if self.trainParam_dictionary['bright_aug'] == 1:
            aug_size = 2 * nbr_anom_train_patches
        if self.trainParam_dictionary['bright_aug'] == 0:
            aug_size = nbr_anom_train_patches

        if self.verbose:
            print("    Augmenting rotation of images")
        rotations = np.random.randint(low=1, high=4, size=aug_size).astype("int32")
        counter2 = tf.data.Dataset.from_tensor_slices(rotations)
        train_ds_to_rotate = tf.data.Dataset.zip((train_ds_anomaly, counter2))
        train_ds_rotated = train_ds_to_rotate.map(rotate, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if self.verbose:
            print("    Augmenting flip of images")

        flip_seeds = np.random.randint(low=1, high=11, size=aug_size).astype("int32")
        counter3 = tf.data.Dataset.from_tensor_slices(flip_seeds)
        train_ds_to_flip = tf.data.Dataset.zip((train_ds_anomaly, counter3))
        train_ds_flipped = train_ds_to_flip.map(flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        train_ds_rotated_flipped = train_ds_anomaly.concatenate(train_ds_rotated).concatenate(train_ds_flipped)

        augmented = train_ds_rotated_flipped
        anomalous_train = aug_size * 3

        normal_train = self.trainParam_dictionary['normal_to_anom'] * anomalous_train
        frac = self.trainParam_dictionary['normal_to_anom']
        if self.verbose:
            print("    Number of normal patches: %s, number of anomalous patches: %s"% (normal_train, anomalous_train))
        if self.verbose:
            print("    Normal patches per anomalous patch in training data: %s" % frac)

        normal_train_ds = normal_train_ds.shuffle(normal_train * 5).take(normal_train)
        val_ds_final = val_ds_anomaly.concatenate(normal_val_ds.take(nbr_anom_val_patches * frac))
        train_ds_final = normal_train_ds.concatenate(augmented)

        train_ds_final = train_ds_final.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        val_ds_final = val_ds_final.map(format_data, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        train_ds_final = train_ds_final.shuffle(buffer_size=normal_train + anomalous_train, reshuffle_each_iteration=True)

        train_ds_batch = train_ds_final.batch(batch_size=self.trainParam_dictionary['batch_size'], drop_remainder=True)
        val_ds_batch = val_ds_final.batch(batch_size=self.trainParam_dictionary['batch_size'], drop_remainder=False)

        time2 = time.time()
        pro_time = time2 - time1
        if self.verbose:
            print("Training and validation datasets created (processing time was {:.2f} s), starting training...".format(pro_time))
            print()
            print("--- TRAINING: Training ---")

        checkpoint_callback_best_loss = tf.keras.callbacks.ModelCheckpoint(filepath=self.model_saving_filepath, monitor="val_loss", mode="min", verbose=1, save_best_only=True)
        #train_param_callback = ModelSavingCallback(filepath=self.model_saving_filepath)
        neptune_cbk = NeptuneCallback(run=self.run, base_namespace="metrics")

        if self.verbose: print("Starting training:")
        self.classifier.fit(train_ds_batch.prefetch(1), epochs=self.trainParam_dictionary['epochs'], initial_epoch=self.trainParam_dictionary['current_epoch'], validation_data=val_ds_batch.prefetch(1), callbacks=[checkpoint_callback_best_loss,neptune_cbk],)
        if self.verbose: print("Training finished.")

    @tf.function
    def process_crop_encode(self, image, label):
        image, label = crop(image, label)
        image, label = encode(image, label, self.ae)
        return image, label

    @tf.function
    def process_crop(self, image, label):
        image, label = crop(image, label)
        return image, label

    def plot_cm(self, labels, predictions, p, label):
        x_labels = ["False Positive", "False Negative"]
        y_labels = ["True Positive", "True Negative"]
        cm = confusion_matrix(labels, predictions, normalize="true")
        heatmap = sns.heatmap(cm, annot=True)
        figure = heatmap.get_figure()
        plt.title(label)
        plt.savefig(os.path.join(self.saving_path, "cm_%s.png" % label), dpi=DPI)
        self.run[f'test_images/cm_%s' % label].upload(figure)
        plt.close(figure)
        #cm_display.show()

    def plot_roc(self, name, labels, predictions, title, **kwargs):
        fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
        fig = plt.figure()
        plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
        plt.xlabel("False positives [%]")
        plt.ylabel("True positives [%]")
        plt.xlim([-0.5, 50])
        plt.ylim([50, 100.5])
        plt.grid(True)
        plt.legend(loc="lower right")
        plt.title(title)
        plt.savefig(os.path.join(self.saving_path, "roc_plot.png"), dpi=DPI)
        self.run[f'test_images/roc_plot'].upload(fig)
        plt.close(fig)
        # plt.show()

    def plot_prc(self, name, labels, predictions, title, **kwargs):
        precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)
        fig = plt.figure()
        plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
        plt.xlabel("Precision")
        plt.ylabel("Recall")
        plt.grid(True)
        plt.legend(loc="lower right")
        plt.title(title)
        plt.savefig(os.path.join(self.saving_path, "prc_plot.png"), dpi=DPI)
        self.run[f'test_images/prc_plot'].upload(fig)
        plt.close(fig)
        # plt.show()

    def rounding_thresh(self, input, thresh):
        rounded = np.round(input - thresh + 0.5)
        return rounded

    def plot_histogram(self, labels, pred_flat, p, normalize, ylim):
        true_ano = []
        true_nor = []
        for i, j in zip(labels, pred_flat):
            if i == 0:
                true_nor.append(j)
            if i == 1:
                true_ano.append(j)
        fig = plt.figure()
        plt.grid(zorder = -3)
        plt.xlim(0,1)
        plt.ylim(0,ylim)
        plt.hist(true_nor,bins=int(np.sqrt(len(true_nor))),color="green",density=normalize,alpha=0.7,label="Normal",)
        plt.hist(true_ano,bins=int(np.sqrt(len(true_ano))),color="red",density=normalize,alpha=0.7,label="Anomalous")
        plt.legend()
        plt.savefig(os.path.join(self.saving_path, "pred_hist.png"), dpi=DPI)
        self.run[f'images/histogram_plot'].upload(fig)
        plt.close(fig)
        # plt.show()

    def rounding_thresh(self, input, thresh):
        rounded = np.round(input - thresh + 0.5)
        return rounded

    def plot_thresholds(self, val_labels, val_pred):
        thresholds = np.arange(0.001, 0.4, 0.005)
        val_pred_flat = np.array(val_pred).flatten()
        val_labels = np.array(val_labels).flatten()
        fps = []
        fns = []
        for thresh in thresholds:
            cm = confusion_matrix(val_labels, val_pred_flat > thresh, normalize="true")
            tn, fp, fn, tp = cm.ravel()
            fps.append(fp)
            fns.append(fn)
            self.run['Threshold'].log(thresh)
        fig = plt.figure()
        plt.plot(thresholds, fps, label="FP")
        plt.plot(thresholds, fns, label="FN")
        plt.grid()
        plt.title("FP and FN for the validation set with different classification thresholds")
        plt.xlabel("Threshold")
        plt.ylabel("# of false predictions")
        plt.legend()
        plt.savefig(os.path.join(self.saving_path, "thresholds_plot.png"), dpi=DPI)
        self.run[f'images/val_threshold_plot'].upload(fig)
        plt.close(fig)
        # plt.show()
        return thresholds, fns, fps

    def threshold_from_validation(self, val_ds, N):
        patch_pred, patch_label = [], []
        if self.verbose:
            print("--- TESTING: Starting threshold determination ---")
            print('Results are saved in %s.' % self.saving_path)
        for whole_img, whole_lbl in tqdm(val_ds, total=N):
            ## crop and encode
            whole_img, whole_lbl = crop(whole_img, whole_lbl)
            whole_img_enc, whole_lbl = encode(whole_img, whole_lbl, self.ae)

            ##patch
            whole_img_patched = patch_image(whole_img)
            whole_img_enc_patched = patch_image(whole_img_enc)
            whole_img_patched, whole_lbl = format_data_batch(whole_img_patched, whole_lbl)

            ##predict backgrounds
            br_pred = self.br_model.predict(whole_img_patched)
            br_pred = np.round(br_pred)
            ## get backround as zeroes so acts as a filter
            br_pred_inv = 1 - br_pred
            whole_img_enc_patched, whole_lbl = format_data_batch(whole_img_enc_patched, whole_lbl)

            ## predict labels
            pred = self.classifier.predict(whole_img_enc_patched)
            pred = np.multiply(pred, br_pred_inv)

            whole_lbl = whole_lbl.numpy().flatten()
            patch_pred = np.append(patch_pred, pred.flatten())
            patch_label = np.append(patch_label, whole_lbl)

        ths, fn, fp = self.plot_thresholds(np.array(patch_label).flatten(), patch_pred.flatten())
        th_df = pd.DataFrame(data=np.transpose([ths, fn,fp]), columns=['TH', 'FN', 'FP'])
        limit_fp = 0.001
        th_df = th_df[th_df.FP < limit_fp]
        limit_fn = 0.25
        min_fn = th_df.FN.min()
        print("Minimum of FN with validation set is %s. " % min_fn)
        if min_fn < limit_fn:
            th = th_df.query('FN==%s' % min_fn)['TH'].mean()
            print("Threshold determined based on validation set: %s. " % th)
            return th
        else:
            print("No threshold determined, using default")
            return 0.1

    def create_results_dir(self, title, th, FPR, FNR, f2, acc, precision, recall, tp, tn, fp, fn):
        results = {}
        results['FPR'] = FPR
        results['FNR'] = FNR
        results['f2'] = f2
        results['acc'] = acc
        results['precision'] = precision
        results['recall'] = recall
        results['tp'] = tp
        results['tn'] = tn
        results['fp'] = fp
        results['fn'] = fn
        ser = pd.Series(data=results)

    def evaluate_preds(self, label, prediction, title):
        print("Evaluation results for: %s \n" % title)

        self.plot_cm(label,self.rounding_thresh(prediction, self.th),p = self.th,label="Confusion matrix for %s" % title,)

        cm = confusion_matrix(label, self.rounding_thresh(prediction, self.th))

        if "whole" not in title:
            self.plot_histogram(label, prediction, self.th, True, 10)
        elif "whole" in title:
            self.plot_histogram(label, prediction, self.th, False, 250)

        fl = SigmoidFocalCrossEntropy()
        loss = fl(y_true=label, y_pred=prediction).numpy()
        loss = np.round(loss, 2)
        tn, fp, fn, tp = cm.ravel()
        precision = np.round(tp / (tp + fp), 2)
        recall = np.round(tp / (tp + fn), 2)
        error = np.round((fp + fn) / (tp + tn + fp + fn), 2)
        acc = np.round(1 - error, 2)
        f2 = np.round(sklearn.metrics.fbeta_score(label, self.rounding_thresh(prediction, self.th), beta=2), 2)
        FPR = np.round(fp / (fp + tn), 2)
        FNR = np.round(fn / (fn + tp), 2)

        if self.verbose:
            print("Test loss:     ", loss)
            print("tn, fp, fn, tp:", tn, fp, fn, tp)
            print("Precision:     ", precision)
            print("Recall:        ", recall)
            print("Error:         ", error)
            print("Accuracy:      ", acc)
            print("Fbeta:         ", f2)
            print("FPR:           ", FPR)
            print("FNR:           ", FNR)

        self.run['test of %s/Test loss' % title].log(loss)
        self.run['test of %s/Precision' % title].log(precision)
        self.run['test of %s/Recall' % title].log(recall)
        self.run['test of %s/FPR' % title].log(FPR)
        self.run['test of %s/FNR' % title].log(FNR)

        self.create_results_dir(title, self.th, FPR, FNR, f2, acc, precision, recall, tp, tn, fp, fn)

        if "whole" not in title:
            self.plot_roc("Test",label,prediction,"ROC curve for for %s" % title,linestyle="--")
            self.plot_prc("Test",label,prediction,"PRC curve for for %s" % title,linestyle="--",)
        print()

    def plot_false(self, false_patches, title):
        l = int(len(false_patches) / (PATCHSIZE, PATCHSIZE))
        false_patches = false_patches.reshape(l, PATCHSIZE, PATCHSIZE)
        for i in range(0, l):
            plt.figure(i)
            plt.imshow(false_patches[i].reshape(PATCHSIZE, PATCHSIZE))
            plt.title(title)
            plt.savefig(
                os.path.join(self.saving_path, "F_patch_examples", "F_patch_%s.png" % i), dpi=DPI)
            plt.close(i)
            # plt.show()

    def plot_false_wholes(self, false_wholes, title):
        l = int(len(false_wholes) / (PICTURESIZE_Y * PICTURESIZE_X))
        false_wholes = false_wholes.reshape(l, PICTURESIZE_Y, PICTURESIZE_X)
        for i in range(0, l):
            plt.figure(i)
            false_whole = false_wholes[i].reshape(PICTURESIZE_Y, PICTURESIZE_X)
            plt.imshow(false_whole)
            plt.title(title)
            plt.savefig(os.path.join(self.saving_path, "F_whole_examples", "F_whole_%s.png" % i),dpi=DPI)
            plt.close()
            # plt.show()

    def pred_loop(self, test_ds, test=1, N=500, plot=True, nbr_to_plot=5):
        ind = 0
        whole_pred, whole_label, patch_pred, patch_label = [], [], [], []
        encode_times, prediction_times = [], []
        false_positive_patches, false_negative_patches, false_negative_wholes = [],[],[]
        if self.verbose:
            print("--- TESTING: Starting testing ---")
            print('Results are saved in %s.' % self.saving_path)
        for whole_img, whole_lbl in tqdm(test_ds, total=N):
            ## crop and encode
            whole_img, whole_lbl = crop(whole_img, whole_lbl)
            enc_t1 = time.time()
            whole_img_enc, whole_lbl = encode(whole_img, whole_lbl, self.ae)
            enc_t2 = time.time()
            encode_times.append(enc_t2 - enc_t1)

            ##patch
            whole_img_patched = patch_image(whole_img)
            whole_img_enc_patched = patch_image(whole_img_enc)
            whole_img_patched, whole_lbl = format_data_batch(whole_img_patched, whole_lbl)

            ##predict backgrounds
            br_pred = self.br_model.predict(whole_img_patched)
            br_pred = np.round(br_pred)
            br_pred_ids = np.where(br_pred > self.br_th)[0]
            ## get backround as zeroes so acts as a filter
            br_pred_inv = 1 - br_pred
            whole_img_enc_patched, whole_lbl = format_data_batch(whole_img_enc_patched, whole_lbl)

            ## predict labels
            pred_t1 = time.time()
            pred = self.classifier.predict(whole_img_enc_patched)
            pred_t2 = time.time()
            prediction_times.append(pred_t2 - pred_t1)

            pred = np.multiply(pred, br_pred_inv)
            pred_ids = np.where(pred > self.th)[0]

            whole_lbl = whole_lbl.numpy().flatten()
            lbl_ids = np.where(whole_lbl == 1.0)[0]

            patch_pred = np.append(patch_pred, pred.flatten())
            patch_label = np.append(patch_label, whole_lbl)

            if len(pred_ids) > 0:
                whole_pred_i = 1
            elif len(pred_ids) == 0:
                whole_pred_i = 0
            whole_pred = np.append(whole_pred, whole_pred_i)
            if len(lbl_ids) > 0:
                whole_label_i = 1
            elif len(lbl_ids) == 0:
                whole_label_i = 0
            whole_label = np.append(whole_label, whole_label_i)
            if whole_label_i == 1 and whole_pred_i == 0:
                false_negative_wholes = np.append(false_negative_wholes, whole_img)
            if plot and test == 1:
                # print('Whole label : ', whole_label)
                img, ax = plt.subplots()
                whole_img_to_plot = whole_img.numpy().reshape(PICTURESIZE_Y, PICTURESIZE_X)
                whole_img_to_plot = cv2.cvtColor(whole_img_to_plot.astype("uint8"), cv2.COLOR_BAYER_RG2RGB)
                ax.imshow(whole_img_to_plot)
                for t in lbl_ids:
                    x_t, y_t = box_index_to_coords(t)
                    plt.gca().add_patch(patches.Rectangle((x_t, y_t), PATCHSIZE, PATCHSIZE, linewidth=2, edgecolor="r", facecolor="none"))
                    if t not in pred_ids:
                        false_negative_patches = np.append(false_negative_patches, whole_img_enc_patched[t].numpy().reshape(PATCHSIZE, PATCHSIZE))
                for p in pred_ids:
                    if p in lbl_ids:
                        color = "g"
                    elif p not in lbl_ids:
                        color = "y"
                        false_positive_patches = np.append(false_positive_patches, whole_img_enc_patched[p].numpy().reshape(PATCHSIZE, PATCHSIZE))
                    x_p, y_p = box_index_to_coords(p)
                    plt.gca().add_patch(patches.Rectangle((x_p, y_p), PATCHSIZE, PATCHSIZE, linewidth=2, edgecolor=color, facecolor="none"))
                for b in br_pred_ids:
                    x_b, y_b = box_index_to_coords(b)
                    plt.gca().add_patch(
                        patches.Rectangle(
                            (x_b, y_b),
                            PATCHSIZE,
                            PATCHSIZE,
                            linewidth=2,
                            edgecolor="gray",
                            facecolor="none",
                        )
                    )
                plt.savefig(os.path.join(self.saving_path,"pred_whole_examples","example_plot_%s.png" % ind), dpi=DPI)
                plt.close()
                # plt.show()
            ind = ind + 1
            if ind > nbr_to_plot:
                plot = False
        return (
            np.array(whole_pred).flatten(),
            np.array(whole_label).flatten(),
            patch_pred.flatten(),
            np.array(patch_label).flatten(),
            false_positive_patches,
            false_negative_patches,
            false_negative_wholes,
        )

    def init_testing(self, test_db, val_db):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
        np.random.seed(42)
        self.run = neptune.init(
            name = 'testing',
            with_id = 'HEX-31',
            tags=['testing'],
            project="sgroenro/HexaScanWithAI",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIyOGNiZjUwYy1lZTE4LTQ0YWItODEwMy01ZDYxYmZjM2VjYWUifQ==")

        if self.verbose:
            print()
            print("--- TESTING: Loading models and data ---")
            print()

        #times_normal = self.trainParam_dictionary['normal_to_anom']
        times_normal = 5

        create_dir(os.path.join(self.saving_path, "pred_whole_examples"))
        create_dir(os.path.join(self.saving_path, "F_whole_examples"))
        create_dir(os.path.join(self.saving_path, "F_patch_examples"))

        self.loadBRModel(os.path.join(MODELS_DIR_LOC, "br_cnn_v3"))
        self.ae = AutoEncoder()
        self.ae.load(self.ae_location)

        X_test_det_list, Y_test_det_list = process_anomalous_df_to_numpy(test_db)
        X_val_det_list, Y_val_det_list = process_anomalous_df_to_numpy(val_db)

        X_test_norm_list = np.load(os.path.join(self.train_db_loc, "NORMAL_TEST_20220711.npy"), allow_pickle=True)
        X_val_norm_list = np.load(os.path.join(self.train_db_loc, "NORMAL_VAL_20220711.npy"), allow_pickle=True)

        if self.verbose: print(); print("Available normal whole test images: %s" % len(X_test_norm_list))

        N_det_test = len(X_test_det_list)
        N_det_val = len(X_val_det_list)
        if self.verbose: print('    Loaded number of anomalous whole test images: %s. ' % N_det_test)
        nbr_of_normals = N_det_test * times_normal
        nbr_of_normals_val = N_det_val * times_normal
        if nbr_of_normals < len(X_test_norm_list):
            if self.verbose:
                print('    Sampling %s whole images from normal images for testing.' % nbr_of_normals)
                print('    Sampling %s whole images from normal images for validation. \n' % nbr_of_normals_val)
            np.random.seed(42)
            X_test_norm_list = np.random.choice(X_test_norm_list, nbr_of_normals, replace=False)
            X_val_norm_list = np.random.choice(X_val_norm_list, nbr_of_normals_val, replace=False)
        elif self.verbose:
            print('    Using all normal images. \n')

        X_test_norm_list = [os.path.join(self.images_dir_loc, s) for s in X_test_norm_list]
        X_val_norm_list = [os.path.join(self.images_dir_loc, s) for s in X_val_norm_list]

        X_test_det_list = [os.path.join(self.images_dir_loc, s) for s in X_test_det_list]
        X_val_det_list = [os.path.join(self.images_dir_loc, s) for s in X_val_det_list]

        N_normal_test = len(X_test_norm_list)
        Y_test_normal_list = np.full((N_normal_test, PATCHES), 0.0)
        Y_val_normal_list = np.full((N_normal_test, PATCHES), 0.0)

        test_norm_ds = create_cnn_dataset(X_test_norm_list, Y_test_normal_list.tolist())
        test_anom_ds = create_cnn_dataset(X_test_det_list, Y_test_det_list.tolist())
        test_ds = test_anom_ds.concatenate(test_norm_ds).shuffle(buffer_size=1000).batch(1)

        val_norm_ds = create_cnn_dataset(X_val_norm_list, Y_val_normal_list.tolist())
        val_anom_ds = create_cnn_dataset(X_val_det_list, Y_val_det_list.tolist())
        val_ds = val_anom_ds.concatenate(val_norm_ds).shuffle(buffer_size=1000).batch(1)

        if True:
            self.th = self.threshold_from_validation(val_ds, N = (times_normal+1)*N_det_val)

        if self.verbose:
            print("Testing with classification threshold: %s \n" % self.th)

        test_hyperparameters = {'model_name': self.model_name, 'times_normal': times_normal, 'threshold': self.th,
                                'loss_name': self.trainParam_dictionary['lossFunc']}
        self.run['test_hyperparameters'] = test_hyperparameters

        (whole_pred, whole_label, patch_pred, patch_label, false_positive_patches, false_negative_patches, false_negative_wholes) = self.pred_loop(test_ds = test_ds, test=1, N=N_det_test + N_normal_test)

        # false_positive_patches = np.array(false_positive_patches)
        # false_negative_patches = np.array(false_negative_patches)
        # plot_false(false_positive_patches, 'False POSITIVE')
        # plot_false(false_negative_patches, 'False NEGATIVE')
        # plot_false_wholes(false_negative_wholes, 'False NEGATIVE')

        self.evaluate_preds(patch_label, patch_pred, "test patches")
        self.evaluate_preds(whole_label, whole_pred, "test whole images")
