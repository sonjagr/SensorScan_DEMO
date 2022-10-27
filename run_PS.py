# Created by Sonja Grönroos in July 2022
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

import glob, re
import os
import time
import argparse
import tensorflow as tf
import numpy as np

from utility.fs import create_dir
from tensorflow import keras
from tensorflow_addons.losses import SigmoidFocalCrossEntropy
from classes.ImageHandler_v2 import AIImageAnnotateEvent

np.random.seed(42)
tf.get_logger().setLevel('WARNING')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

check = tf.config.list_physical_devices('GPU')
print('GPU check: ', check)

parser = argparse.ArgumentParser()
parser.add_argument("--SensorGeometry", type=str, help="(input) Sensor geometry (HPK_198ch_8inch, HPK_432ch_8inch).",
                    required=True)
parser.add_argument("--CampaignName", type=str,
                    help="Name of campaign.", required=True)
parser.add_argument("--DUTName", type=str,
                    help="Name of DUT.", required=True)
parser.add_argument("--N_images", type=int, help="Number of scanned images of the xy-scan (0-coarsest, 2-finest).", required=True)
parser.add_argument("--Grid", type=int, help="How many grids are applied to an image during inference (1-only default, 2-default+secondary).", required=False, default = 1)
parser.add_argument("--Threshold", type=float, help="(option) Classification threshold (0.-1.).",
                    default=0.1, required=False)
parser.add_argument("--Verbose", type=bool, help="(option) Verbosity: False no printing, True printing.",
                    default=False, required=False)
args = parser.parse_args()

GEOMETRY = args.SensorGeometry
N_images = args.N_images
campaign = args.CampaignName
dut = args.DUTName
grid = args.Grid
ClassifierDetectionThreshold = args.Threshold
verbose = args.Verbose

output_dir = os.path.abspath(os.path.join("outputs", campaign, dut))

print("Analysing DUT %s in campaign %s." % (dut, campaign))
print()

ClassifierModel = os.path.abspath("CNNs/class_cnn_clean_fl_epoch_20_2022-07-26")
AEModel = os.path.abspath("CNNs/AE3_v2_277_epochs")
BRModel = os.path.abspath("CNNs/br_cnn_v3")

ImageDisplay = AIImageAnnotateEvent()
ImageDisplay.setVerbosity(verbose)
ImageDisplay.loadClassifierModel(ClassifierModel)
ImageDisplay.setThreshold(ClassifierDetectionThreshold)
ImageDisplay.loadAE(AEModel)
ImageDisplay.loadBRModel(BRModel)
ImageDisplay.setAI(True)
ImageDisplay.set_DUT(dut)
ImageDisplay.setGrid(grid)
ImageDisplay.set_Campaign(campaign)

skip = False
for _scan_step in range(1, N_images+1):
    time_start_looking = time.time()
    if verbose == 1: print('Step %s:' % _scan_step)
    while True:
        file_to_process = glob.glob(os.path.join(output_dir, 'init*step'+str(_scan_step)+'.npy'))
        # look for if the new step has been completed, if so analyse
        if len(file_to_process) > 0:
            skip = False
            break
        elif (time.time()-time_start_looking) > 60:
            skip = True
            break
        else:
            skip = False
            time.sleep(1)
            continue
    if skip:
        if verbose == 1:
            print("PS analysis on scan step %s was skipped!" % _scan_step)
        continue
    _pad = int(re.search('pad(.*)_step', file_to_process[0]).group(1))
    image_array = np.load(file_to_process[0])
    ImageDisplay.set_img(image_array)
    ImageDisplay.set_filename(file_to_process[0].split('\\')[-1].split('.')[0])
    ImageDisplay.setFilePath(output_dir)
    ImageDisplay.action(_scan_step, _pad)
    rem_rn = np.random.uniform()
    if rem_rn > 0.3:
        os.remove(file_to_process[0])
        if verbose == 1:
            print("Preliminary deletion of initial scan image at step %s." % _scan_step)
    elif rem_rn <= 0.3:
        ImageDisplay.appendSavedStep(_scan_step)
    ann_f = open(os.path.join(output_dir, "annotation_indices.txt"), 'w')
    ann_f.write(str(ImageDisplay.anomalous_scan_steps))
    ann_f.close()

if verbose == 1:
    print("Pre-selection finished. %s images were annotated." % ImageDisplay.count)
    print("Anomalous scan indices: ")
    print(ImageDisplay.anomalous_scan_steps)
    print()

print("All saved scan steps: ", ImageDisplay.saved_scan_steps)
## removing 90 % of normal initial scan npys
normal_saved_scan_steps = np.array([i for i in ImageDisplay.saved_scan_steps if i not in ImageDisplay.anomalous_scan_steps_npy])
anomalous_saved_scan_steps = np.array([i for i in ImageDisplay.saved_scan_steps if i in ImageDisplay.anomalous_scan_steps_npy])

## remove initial scans that were annotated
if len(anomalous_saved_scan_steps) > 0:
    for _ss in anomalous_saved_scan_steps:
        file_to_remove = glob.glob(os.path.join(output_dir, 'init*step' + str(int(_ss)) + '.npy'))
        print("Removing ", file_to_remove)
        os.remove(file_to_remove[0])

print("Of those not annotated/normal: ", normal_saved_scan_steps)
normal_validation_size = int(0.1 * (N_images-ImageDisplay.count))
if normal_validation_size == 0:
    normal_validation_size = 1
print("Normals validation size: " , normal_validation_size)

if normal_validation_size >= len(normal_saved_scan_steps):
    print("No normals to remove.")
elif normal_validation_size < len(normal_saved_scan_steps):
    remove_size = len(normal_saved_scan_steps) - normal_validation_size

    normal_steps_to_remove = np.random.choice(a=list(normal_saved_scan_steps), size=remove_size, replace=False)
    print("Normal scan steps that will be removed:", normal_steps_to_remove)
    normal_saved_scan_steps = [i for i in normal_saved_scan_steps if i not in normal_steps_to_remove]
    print(normal_saved_scan_steps)
    if verbose == 1:
        print("Normal scan indices that were saved for validation: ")
        print(normal_saved_scan_steps)
        print()

    for _ss in normal_steps_to_remove:
        file_to_remove = glob.glob(os.path.join(output_dir, 'init*step' + str(int(_ss)) + '.npy'))
        print("Removing ", file_to_remove)
        os.remove(file_to_remove[0])

