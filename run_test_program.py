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
import time
import glob
import numpy as np

from utility.fs import create_dir, read_AI_indices
from matplotlib import pyplot as plt
from classes.ScanMapping import ScanMapping
from classes.ImageHandler_v2 import ImageValidateEvent, ImageSaveEvent

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

GEOMETRY = 'HPK_198ch_8inch'
campaign = 'dummy_images'
dut = '8inch_198ch_N4790_12'

output_dir = create_dir(os.path.abspath(os.path.join("outputs", campaign, dut)))
jpeg_output_dir = create_dir(os.path.abspath(os.path.join(output_dir, 'jpegs')))
dummy_images = glob.glob(os.path.join(r'backup/dummy_images/8inch_198ch_N4790_12', 'initial*.npy'))
f = open(os.path.join(output_dir, "annotation_indices.txt"), 'a+')

N_images = len(dummy_images)

print(" ----- 1-5 ----- ")
print("THIS IS A TEST PROGRAM. DOES NOT INCLUDE CAMERA INPUT. USES DUMMY INPUT DATA. ONLY %s SCAN INDICES." % N_images)
print(N_images, ' images will be taken.')

TH = 0.2
DL = 0
sm = ScanMapping(GEOMETRY)
sm.loadGeoFile()
sm.createPattern(detail_level=DL)
sm.initialise_scan()
sm.openFigure()
fig = sm.visualisePattern()
print()

print()
print(" ----- 5: Test photo ----- ")
msg = "Please place the central pad underneath the microscope. \n"
msg += "And please hit 'ENTER' afterwards. \n"
input(msg)

i = 0
while True:
    if i == 0:
        print('Taking photos: adjust sensor placement. Press enter to quit. \n', flush=True)
        ans = input()
        break
    time.sleep(0.2)
    i = 1
print()

print(" ----- 6: Set exposure ----- ")

while True:
    current_exposure_time = 0
    msg = "By what percentage to scale [-100., 100]? \n"
    msg += "Type '0' to skip. \n"
    try:
        scaling = float(input(msg))
    except:
        print("Input must be a number.")
        continue
    if scaling == 0:
        break
    new_exposure_time = current_exposure_time * max(0, 1. + scaling / 100.)
print()

# initial scan images are saved to their own folder - in the future this will not be necessary
ImageSave = ImageSaveEvent()
ImageSave.setBasePath(os.path.join(jpeg_output_dir, "initial_scan.jpeg"))
ImageSave.setJPEGQuality(10)

# npy arrays are saved in the folder where PS analysis can read them from.
# 90 % of npy arrays for images deemed clean are removed.
ImageSave2 = ImageSaveEvent()
ImageSave2.setBasePath(os.path.join(output_dir, "initial_scan.npy"))

print(" ----- 9: Scan start ----- ")
input("Please hit 'ENTER' to initiate the wafer scan.")
first_scan = True
anomalous_scan_steps_npy = []
for i in range(1, 20+1):
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
            cmd = "python ./run_PS.py --N_images %s --SensorGeometry %s --Threshold %s --CampaignName %s --DUTName %s --Verbose True --Grid %s" % (N_images, GEOMETRY, TH, campaign, dut, '2')
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

print("Waiting for PS to finish... \n")
while True:
    myProcessIsRunning = AI_process.poll() is None
    if not myProcessIsRunning:
        break
    else:
        continue
print("...PS has finished. \n")

sm.update_title("Scan map with pre-selected anomalies")
AI_scan_steps = read_AI_indices(output_dir)
for i in AI_scan_steps:
    print(i)
    sm.setNAnnotations(i, 1)

print("PS annotated %s images out of %s. \n" % (len(AI_scan_steps), N_images))

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
    print("Skipping validation: no annotations!")
elif len(AI_scan_steps) > 0:
    while True:
        validate = input("Will you validate the images annotated by the PS now? Validation step not mandatory during pre-series testing. (yes/no) \n")
        if validate.casefold() == "yes".casefold():
            validation.validation_action(type='all', overwrite=False)
            validation.createValidationSummary()
            break
        elif validate.casefold() == "no".casefold():
            print("Remember to run validation_tool.py later. \n")
            break
        else:
            print("Try again \n")
            continue
plt.close(2)
for i in AI_scan_steps:
    if i not in validation.anomalous_scan_steps_npy:
        sm.update_title("Scan map with validated anomalies")
        sm.setNAnnotations(i, 0)
plt.close(2)
for i in validation.anomalous_scan_steps_npy:
    if i not in AI_scan_steps:
        sm.update_title("Scan map with validated anomalies")
        sm.update_legend()
        sm.setNAnnotations(i, 2)

print()
print(" ----- 11: Visual inspection ----- ")

anomalous_scan_steps = validation.anomalous_scan_steps_npy
ImageSave3 = ImageSaveEvent()
ImageSave3.setBasePath(os.path.join(jpeg_output_dir, "rescan.jpeg"))

print("Revisiting suspicious areas \n")
sm.update_title("Scan map during visual inspection")

if len(anomalous_scan_steps) == 0 or anomalous_scan_steps is None:
    print("  PS has no suggestions. \n")
elif len(anomalous_scan_steps) > 0 and anomalous_scan_steps is not None:
    print("PS suggests you visit these step indexes: %s \n" % anomalous_scan_steps)
for i in anomalous_scan_steps:
    answer = input("Press enter to move to the next anomalous step: %s. Type 'END' to stop. \n" % i)
    if answer.casefold() == "END".casefold():
        break
    else:
        try:
            step = int(i)
        except:
            print("Input must be an integer.")
            continue
        _exists, current_pad_nr, _x, _y = sm.scan_step_coordinates(step)
        if not _exists:
            continue
        while True:
            answer = input("Take video at scan index %s to show cleaning? (yes/no) \n" % step)
            if answer.casefold() == 'END'.casefold():
                break
            if answer.casefold() == "yes".casefold():
                i = 0
                while True:
                    if i == 0:
                        ans = input('Taking photos. Press enter to quit cleaning: a rescan will be taken and saved of the cleaned area. \n', flush=True)
                        break
                    time.sleep(0.1)
                    i = 1
                plt.close(2)
                ImageSave2.setPathPostfix("pad%i_step%i" % (current_pad_nr, step))
                plt.close(2)
                break
            elif answer == "no":
                break
            else:
                continue
        sm.setNAnnotations(step, 0)

print()
print(" ----- 12: End. ----- ")
print("End of the scanning program.")
print("Please manually clean the guard ring area now using the microscope and joystick.")

