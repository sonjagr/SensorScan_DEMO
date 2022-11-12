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

import os, sys, subprocess
import msvcrt, time
import glob
import numpy as np
import cv2
import argparse

from tqdm import tqdm
from utility.fs import create_dir, read_AI_indices
from matplotlib import pyplot as plt
from classes.ScanMapping import ScanMapping
from classes.ImageHandler_v2 import ImageValidateEvent, ImageSaveEvent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument("--SensorGeometry", type=str, help="Mapping geometry, see options in folder /maps. Usually custom_map_xxx, default = custom_map_LD_Full_385.",
                    required=False, default ="custom_map_LD_Full_385")
parser.add_argument("--PS", type=int, help="(option) Use pre-selection (1) or not (0). Default = 1.", default=1, required=False)
parser.add_argument("--COMPORT", type=int, help="(option) COM port of the xy stage. Default = 3.", default=3, required=False)
parser.add_argument("--Threshold", type=float, help="(option) Classification threshold (0.-1.).",
                    default=0.2, required=False)
parser.add_argument("--Grid", type=int, help="(option) How many grids are applied to an image during inference (1-only default, 2-default+secondary).", required=False, default = 1)
args = parser.parse_args()

GEOMETRY = args.SensorGeometry
TH = args.Threshold
if TH > 1. or TH < 0:
    print("Incorrect value for argument 'Threshold'. Try again.")
    exit()
grid = args.Grid
if grid > 2 or grid < 1:
    print("Incorrect value for argument 'Grid'. Try again.")
    exit()
PS = args.PS
if PS == 0:
    PS = False
elif PS == 1:
    PS = True
else:
    print("Incorrect value for argument 'PS'. Try again.")
    exit()

campaign = 'dummy_images'
dut = '8inch_198ch_N4790_12'

output_dir = create_dir(os.path.abspath(os.path.join("outputs", campaign, dut)))
jpeg_output_dir = create_dir(os.path.abspath(os.path.join(output_dir, 'jpegs')))
dummy_images = glob.glob(os.path.join(r'backup/dummy_images/8inch_198ch_N4790_12', 'initial*.npy'))

print(" ----- 1-7 ----- ")
print("THIS IS A TEST PROGRAM. DOES NOT INCLUDE CAMERA INPUT: USES DUMMY INPUT DATA. ONLY %s SCAN INDICES." % len(dummy_images))

if not PS:
    print("Running without pre-selection!")
elif PS:
    print("Running with pre-selection!")

print(" ----- 8: Loading ----- ")
print("Loading the scan mapping")
try:
    sm = ScanMapping(GEOMETRY)
except:
    print("Scan map name does not exist.")
    exit()
sm.loadGeoFile()
if "custom" in GEOMETRY:
    print("You have chosen a custom mapping named %s." % GEOMETRY)
    print("Detail level for custom maps is 0.")
    DL = 0
    change = False
elif GEOMETRY == "HPK_198ch_8inch":
    DL = 2
    print("Default detail level for the mapping of LD sensor is 2 (529 images).")
    change = True
elif GEOMETRY == "HPK_432ch_8inch":
    DL = 1
    print("Default detail level for the mapping of HD sensor is 1 (864 images).")
    change = True
if change:
    while True:
        answer = input("Will you change detail level for scan (no/[0,1,2])?\n")
        if answer.casefold() == 'no'.casefold():
            break
        elif (answer == '0') or (answer == '1') or (answer == '2'):
            DL = int(answer)
            print("Detail level is set to ", answer)
            break
        else:
            print("Try again")
            continue
sm.createPattern(detail_level=DL)

sm.initialise_scan()
N_points = sm.N_scan_points
print(N_points, ' images in scan map.')
N_images = len(dummy_images)
print(N_images, ' images will be taken.')
sm.visualisePattern()
print()

f = open(os.path.join(output_dir, "annotation_indices.txt"), 'a+')

ImageSave = ImageSaveEvent()
ImageSave.setBasePath(os.path.join(jpeg_output_dir, "initial_scan.jpeg"))
ImageSave.setJPEGQuality(10)

ImageSave2 = ImageSaveEvent()
ImageSave2.setBasePath(os.path.join(output_dir, "initial_scan.npy"))

print(" ----- 9: Scan start ----- ")
print("Make sure that the light cone/upper light is OFF!")
print("And that the lower light is at MAXIMUM.")
input("Please press ENTER to initiate the wafer scan.")

first_scan = True
anomalous_scan_steps_npy = []

for i in range(1, N_images+1):
    _ = sm.next_xy()
    if _ == -1:
        break
    else:
        scan_step = i
        current_pad_nr = _[1]
        ImageSave.setPathPostfix("pad%i_step%i" % (current_pad_nr, scan_step))
        ImageSave2.setPathPostfix("pad%i_step%i" % (current_pad_nr, scan_step))
        _x = _[2]
        _y = _[3]
        array = np.load(dummy_images[scan_step-1])
        np.save(os.path.join(output_dir, "initial_scan_pad%i_step%i" % (current_pad_nr, scan_step)), array)
        sys.stdout.write("\r{0}".format("Image %s taken.    " % scan_step))
        sys.stdout.flush()
        if first_scan:
            print("Starting pre-selection. \n")
            print("Ignore tensorflow WARNING messages: \n")
            prints_file = open(os.path.join(output_dir, "run_PS_prints.txt"), 'a')
            cmd = "python ./run_PS.py --N_images %s --SensorGeometry %s --Threshold %s --CampaignName %s --DUTName %s --Verbose True --Grid %s" % (N_images, GEOMETRY, TH, campaign, dut, grid)
            AI_process = subprocess.Popen(cmd, stdout=prints_file, stderr=prints_file, shell=True)
            first_scan = False
        sm.setNAnnotations(scan_step, 0)
        if scan_step % 5 == 0 or scan_step == N_images:
            AI_scan_steps = read_AI_indices(output_dir)
            if len(AI_scan_steps) > 0:
                for i in AI_scan_steps:
                    if i not in anomalous_scan_steps_npy and i < scan_step:
                        sm.setNAnnotations(i, 1)
            anomalous_scan_steps_npy = AI_scan_steps
        if scan_step == N_images:
            break
        time.sleep(1)

print()
print("\nScan finished.")
print()

print("Waiting for pre-selection to finish... \n")
while True:
    myProcessIsRunning = AI_process.poll() is None
    if not myProcessIsRunning:
        break
    else:
        continue
print("...pre-selection has finished. \n")

AI_scan_steps = read_AI_indices(output_dir)
for i in AI_scan_steps:
    print(i)
    sm.setNAnnotations(i, 1)

percent = np.round((len(AI_scan_steps) / N_images) * 100, 2)
print("%s images out of %s (%s %%) were pre-selected to be anomalous. \n" % (len(AI_scan_steps), percent, N_images))

print(" ----- 10: Validation ----- ")
validation = ImageValidateEvent()
validation.set_DUT(dut)
validation.set_Campaign(campaign)
validation.setFilePath(output_dir)
validation.set_frac(1)
validation.setAI(0)
validation.anomalous_scan_steps_npy = AI_scan_steps

overwrite = True
if len(AI_scan_steps) == 0:
    print("No pre-selected images!")
elif len(AI_scan_steps) > 0:
    while True:
        validate = input("Press ENTER to validate the pre-selected images that contain annotations and a fraction of normal images. Type in 'no' to skip.\n")
        if validate.casefold() == "no".casefold():
            print("Remember to run validation_tool.py later. \n")
            break
        else:
            validation.validation_action(type='all', overwrite=False)
            validation.createValidationSummary()
            break
plt.close(2)
for i in AI_scan_steps:
    if i not in validation.anomalous_scan_steps_npy:
        sm.setNAnnotations(i, 0)
plt.close(2)
for i in validation.anomalous_scan_steps_npy:
    if i not in AI_scan_steps:
        sm.update_legend()
        sm.setNAnnotations(i, 2)

print()
print(" ----- 11: Visual inspection ----- ")

anomalous_scan_steps = validation.anomalous_scan_steps_npy
ImageSave3 = ImageSaveEvent()
ImageSave3.setBasePath(os.path.join(jpeg_output_dir, "rescan.jpeg"))

print("Revisiting suspicious areas \n")

if len(anomalous_scan_steps) == 0 or anomalous_scan_steps is None:
    print(" No pre-selected suggestions.  \n")
elif len(anomalous_scan_steps) > 0 and anomalous_scan_steps is not None:
    print(" Pre-selection suggests you visit one of these step indexes: %s \n" % anomalous_scan_steps)

sm.visualisePattern()
print("YOU ARE RUNNING A TEST PROGRAM: VISUAL INSPECTION SKIPPED")
print()
print(" ----- 12: End. ----- ")
print("End of the scanning program.")

while True:
    if GEOMETRY.find("custom") != -1:
        answer = input(
            "With this custom map, it is possible to now go through images of the guard ring on screen. Type in 'no' to skip, otherwise, press ENTER. \n")
        if answer != "no":
            print("YOU ARE RUNNING A TEST PROGRAM: GUARD RING INSPECTION SKIPPED")
            break
        elif answer == "no":
            print("Please manually clean the guard ring area now using the microscope and joystick.")
            break
        else:
            continue
    else:
        print("Please manually clean the guard ring area now using the microscope and joystick.")
        break