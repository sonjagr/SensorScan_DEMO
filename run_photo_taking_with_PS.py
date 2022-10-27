# original author: Thorben Quast
# Full sensor scanning, with AI-based feature detection.
# Modified by Sonja Grönroos in July 2022

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
import argparse
import sys
import subprocess
import numpy as np
import glob
import cv2
import msvcrt, time

from config import *
from matplotlib import pyplot as plt
from tqdm import tqdm
from ctypes import WinDLL
from utility.fs import create_dir, read_AI_indices

from classes.StageController import StageController
from classes.LEICAController import LEICAController
from classes.ImageHandler import VerboseImageEvent, ImageShowEvent, ImageSaveEvent
from classes.ImageHandler_v2 import ImageValidateEvent
from classes.ScanMapping import ScanMapping
#from classes.PrintLogger import Logger

np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--SensorGeometry", type=str, help="Mapping geometry (HPK_198ch_8inch, HPK_432ch_8inch, custom_map_LD_396, custom_map_HD_396, custom_map_LD_385, custom_map_HD_385).",
                    required=False, default ="custom_map_LD_385")
parser.add_argument("--CampaignName", type=str,
                    help="Name of campaign.", required=True)
parser.add_argument("--DUTName", type=str,
                    help="(option) Name of DUT. Default = temp_DUT_name.", required=False, default="temp_DUT_name")
parser.add_argument("--PS", type=int, help="(option) Use pre-selection (1) or not (0). Default = 1.", default=1, required=False)
parser.add_argument("--COMPORT", type=int, help="(option) COM port of the xy stage. Default = 3.", default=3, required=False)
parser.add_argument("--Threshold", type=float, help="(option) Classification threshold (0.-1.).",
                    default=0.1, required=False)
parser.add_argument("--Grid", type=int, help="(option) How many grids are applied to an image during inference (1-only default, 2-default+secondary).", required=False, default = 1)
args = parser.parse_args()

GEOMETRY = args.SensorGeometry
TH = args.Threshold
grid = args.Grid
PS = args.PS
if PS == 0:
    PS = False
elif PS == 0:
    PS = True

campaign = args.CampaignName
dut = args.DUTName
# COMPort for communication to ProScan III xy stage controller
COMPORT = args.COMPORT

# indicate the path of the DLL to control the stage SW
dll_path = os.path.join(os.getcwd(), PRIOR_STAGE_DLL)

if __name__ == "__main__":
    if os.path.exists(dll_path):
        SDKPrior = WinDLL(dll_path)
    else:
        raise RuntimeError("DLL to control the xy stage could not be found.")

    if not PS:
        print("Running without pre-selection!")
    elif PS:
        print("Running with pre-selection!")

    print(" ----- 1: Initialize xy-stage ----- ")
    print("Initialising the xy-stage")
    sc = StageController(SDKPrior)
    sc.initialise_sequence()
    sc.APITest()
    sc.connectToPort(COMPORT)
    sc.getSerialNumber()
    sc.setManualMode()
    print("The xy-stage can be manually controlled now.")
    print()

    print(" ----- 2: Self-calibration ----- ")
    msg = "The xy-stage will perform a self-calibration next. \n"
    msg += "Please type 'no' to not calibrate. \n"
    msg += "Otherwise, please remove any DUT from the stage. \n"
    answer = input(msg)
    if not answer.casefold() == "no".casefold():
        sc.setTrueOrigin()

    print()
    print(" ----- 3: Connect ----- ")
    print("Opening the camera...")
    # set up the camera
    lc = LEICAController()
    print("The connected cameras are:")
    lc.showAllConnectedDevices()
    print("Connecting...")
    lc.connectCamera()
    print(lc.getConnectedModelName())
    lc.setImageHandler(VerboseImageEvent())
    print("Starting the communication...")
    lc.startCommunication()
    print("Taking a test photo...")
    lc.takePhoto()
    print()

    print(" ----- 4: Place sensor ----- ")
    msg = "We are about to place sensor. \n"
    msg += "The stage will now move to 0,0. \n"
    msg += "Please type 'no' to stay in this position, press ENTER to continue. \n"
    answer = input(msg)
    if not answer.casefold() == "no".casefold():
        sc.goto(0, 0)
    print()

    print("Opening the image display...")
    ImageDisplay = ImageShowEvent()
    ImageDisplay.setColorMap(None)  # this accepts any of matplotlib's uniform sequential colormaps
    lc.setImageHandler(ImageDisplay)

    print()
    print(" ----- 5: Test photo ----- ")
    msg = "Please place the central pad underneath the microscope. \n"
    msg += "And please hit ENTER afterwards. \n"
    input(msg)
    lc.takePhoto()
    i = 0
    while True:
        lc.takePhoto()
        if i == 0:
            print('Taking photos: adjust sensor placement. Press any key to quit. \n', flush=True)
        if msvcrt.kbhit():
            msvcrt.getch()
            break
        time.sleep(0.5)
        i = 1
    print()

    print(" ----- 6: Set exposure ----- ")
    print("The current, and recommended, exposure time is", lc.getExposureTime(), "ms")

    while True:
        current_exposure_time = lc.getExposureTime()
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
        lc.setExposureTime(new_exposure_time)
        print("Exposure is", lc.getExposureTime(), "ms")
        lc.takePhoto()
    print()

    print(" ----- 7: Scratch pad reading ----- ")
    if dut == "temp_DUT_name":
        answer = input("Stage will move to a position where scratch pad should be visible. Press ENTER to continue. You can type in 'no' to skip: you will then manually look for scratch pad. \n")
        if answer.casefold() == 'no'.casefold():
            move = False
            print("Entering manual control of xy-stage. Please find and read scratch pad. Do not move sensor!")
            sc.setManualMode()
            while True:
                answer = input("Take a photo (yes/no)? \n")
                if answer.casefold() == "yes".casefold():
                    lc.takePhoto()
                    break
                elif answer.casefold() == "no".casefold():
                    break
                else:
                    continue
            dut = input("Give name for DUT. \n")
            plt.close(2)
        else:
            move = True
        if move:
            sc.goto(-2.46, -80.18) #0.16, -77.28
            lc.takePhoto()
            while True:
                answer = input("Is scratch pad visible in image (yes/no)? \n")
                if answer.casefold() == "yes".casefold():
                    dut = input("Give name for DUT. \n")
                    break
                elif answer.casefold() == "no".casefold():
                    print("Manual control of xy-stage is on. Please adjust table/zoom and read scratch pad. Do not move sensor!")
                    sc.setManualMode()
                    i = 0
                    while True:
                        lc.takePhoto()
                        if i == 0:
                            print('Taking photos. Press any key to quit. \n', flush=True)
                            print('You can adjust sharpness of image. Do not move sensor, only the xy-stage. \n', flush=True)
                        if msvcrt.kbhit():
                            msvcrt.getch()
                            break
                        time.sleep(0.5)
                        i = 1
                    dut = input("Give name for DUT. \n")
                    plt.close(2)
                    break
                else:
                    continue
    print("The DUT is named as %s" % dut)
    output_dir = create_dir(os.path.abspath(os.path.join("outputs", campaign, dut)))
    jpeg_output_dir = create_dir(os.path.abspath(os.path.join(output_dir, 'jpegs')))
    d_f = open(os.path.join(output_dir, "DUT_name.txt"), 'w')
    d_f.write(str(dut))
    d_f.close()
    print("The stage will now move to 0,0. \n")
    sc.goto(0, 0)

    ImageDisplay.close()
    print()


    #sys.stdout = Logger(dir = output_dir)

    print(" ----- 8: Loading ----- ")
    print("Loading the scan mapping")
    sm = ScanMapping(GEOMETRY)
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
    N_images = sm.N_scan_points
    print(N_images, ' images will be taken.')
    sm.visualisePattern()
    print()

    if PS:
        f = open(os.path.join(output_dir, "annotation_indices.txt"), 'a+')

    ImageSave = ImageSaveEvent()
    lc.setImageHandler(ImageSave)
    ImageSave.setBasePath(os.path.join(jpeg_output_dir, "initial_scan.jpeg"))
    ImageSave.setJPEGQuality(10)

    ImageSave2 = ImageSaveEvent()
    lc.appendImageHandler(ImageSave2)
    ImageSave2.setBasePath(os.path.join(output_dir, "initial_scan.npy"))

    print(" ----- 9: Scan start ----- ")
    print("Make sure that the light cone/upper light is OFF!")
    input("Please press ENTER to initiate the wafer scan.")
    first_scan = True
    anomalous_scan_steps_npy = []

    sc.setScanMode()
    #N_images = 20 #for testing purposes
    while True:
        _ = sm.next_xy()
        if _ == -1:
            break
        else:
            scan_step = _[0]
            current_pad_nr = _[1]
            ImageSave.setPathPostfix("pad%i_step%i" % (current_pad_nr, scan_step))
            ImageSave2.setPathPostfix("pad%i_step%i" % (current_pad_nr, scan_step))
            _x = _[2]
            _y = _[3]
            sc.goto(_x, _y)
            lc.takePhoto()
            sm.setNAnnotations(scan_step, 0)
            if PS and first_scan:
                print("Starting the pre-selection (PS) process. \n")
                print("Ignore tensorflow WARNING messages: \n")
                prints_file = open(os.path.join(output_dir, "run_PS_prints.txt"), 'a')
                cmd = "python ./run_PS.py --N_images %s --SensorGeometry %s --Threshold %s --CampaignName %s --DUTName %s --Verbose True --Grid %s" % (N_images, GEOMETRY, TH, campaign, dut, grid)
                AI_process = subprocess.Popen(cmd, stdout=prints_file, stderr=prints_file, shell=True)
                first_scan = False
            if PS:
                if scan_step % 5 == 0 or scan_step == N_images:
                    AI_scan_steps = read_AI_indices(output_dir)
                    if len(AI_scan_steps) > 0:
                        for i in AI_scan_steps:
                            if i not in anomalous_scan_steps_npy and i <= scan_step:
                                sm.setNAnnotations(i, 1)
                    anomalous_scan_steps_npy = AI_scan_steps
            if scan_step == N_images:
                break

    print()
    print("\nScan finished.")
    print()

    if PS:
        print("Waiting for pre-selection to finish... \n")
        while True:
            myProcessIsRunning = AI_process.poll() is None
            if not myProcessIsRunning:
                break
            else:
                #list_of_files = glob.glob(os.path.join(output_dir, 'ann*.npy'))
                #newest_file = max(list_of_files, key=os.path.getctime)
                #newest_file_step = newest_file.split("step")[-1].split(".")[0]
                #print("Pre-selection is at scan step %s/%s." % (newest_file_step, N_images))
                continue
        print("...pre-selection has finished. \n")

        #sm.update_title("Scan map with pre-selected anomalies")
        AI_scan_steps = read_AI_indices(output_dir)
        for i in AI_scan_steps:
            sm.setNAnnotations(i, 1)

        percent = np.round((len(AI_scan_steps)/N_images)*100, 2)
        print("%s images out of %s (%s %%) were pre-selected to be anomalous. \n" % (len(AI_scan_steps), percent, N_images))

        print(" ----- 10: Validation ----- ")
        validation = ImageValidateEvent()
        validation.set_DUT(dut)
        validation.set_Campaign(campaign)
        validation.setFilePath(output_dir)
        validation.set_frac(1)
        validation.setAI(0)
        validation.anomalous_scan_steps_npy = AI_scan_steps

        if len(AI_scan_steps) == 0:
            print("No pre-selected images!")
        elif len(AI_scan_steps) > 0:
            overwrite = False
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
        ## update scan map with false positives
        for i in AI_scan_steps:
            if i not in validation.anomalous_scan_steps_npy:
                #sm.update_title("Scan map with validated anomalies")
                sm.setNAnnotations(i, 0)
        ## update scan map with false negatives
        for i in validation.anomalous_scan_steps_npy:
            if i not in AI_scan_steps:
                #sm.update_title("Scan map with validated anomalies")
                sm.update_legend()
                sm.setNAnnotations(i, 2)

        print()
        anomalous_scan_steps = validation.anomalous_scan_steps_npy
    elif not PS:
        anomalous_scan_steps = np.arange(0,2000)

    print(" ----- 11: Visual inspection ----- ")
    sc.goto(0, 0)

    print("Revisiting suspicious areas \n")
    #sm.update_title("Scan map during visual inspection")

    if len(anomalous_scan_steps) > 1999 or anomalous_scan_steps is None:
        print("  No pre-selected suggestions. \n")
        PS = False
    elif len(anomalous_scan_steps) > 0 and anomalous_scan_steps is not None:
        print("  Pre-selection suggests you visit one of these step indexes: %s \n" % anomalous_scan_steps)

    for i in anomalous_scan_steps:
        if PS:
            msg = "Press ENTER to move to the next anomalous step: %s. Type 'END' to stop. \n" % i
        elif not PS:
            msg = "Type in step index of sensor area to re-visit. Type 'END' to stop. \n"
        answer = input(msg)
        if answer.casefold() == "END".casefold():
            break
        else:
            try:
                if PS:
                    step = int(i)
                elif not PS:
                    step = int(answer)
            except:
                print("Input must be an integer.")
                continue
            _exists, current_pad_nr, _x, _y = sm.scan_step_coordinates(step)
            if not _exists:
                continue
            sc.goto(_x, _y)

            ImageDisplay_insp = ImageShowEvent()
            ImageDisplay_insp.setColorMap(None)  # this accepts any of matplotlib's uniform sequential colormaps
            lc.setImageHandler(ImageDisplay_insp)

            ImageSave3 = ImageSaveEvent()
            lc.appendImageHandler(ImageSave3)
            ImageSave3.setBasePath(os.path.join(jpeg_output_dir, "rescan.jpeg"))
            ImageSave3.setJPEGQuality(10)

            ImageSave3.setPathPostfix("pad%i_step%i" % (current_pad_nr, step))
            while True:
                #answer = input("Take video at scan index %s to show cleaning (yes/no)?\n" % step)
                answer = 'yes'
                if answer.casefold() == "yes".casefold():
                    i = 0
                    while True:
                        lc.takePhoto()
                        if i == 0:
                            print('Taking photos. Press any key to quit cleaning: a rescan .jpeg will be taken of the cleaned area and saved. \n', flush=True)
                        if msvcrt.kbhit():
                            msvcrt.getch()
                            break
                        time.sleep(0.5)
                        i = 1
                    plt.close(2)
                    sm.setNAnnotations(step, 0)
                    break
                elif answer.casefold() == "no".casefold() or answer.casefold() == "END".casefold():
                    lc.takePhoto()
                    plt.close(2)
                    sm.setNAnnotations(step, 0)
                    break
                else:
                    continue

    sm.visualisePattern()
    print()
    print(" ----- 12: End. ----- ")
    print("End of the scanning program.")

    while True:
        if GEOMETRY == "custom_map_HD_385" or GEOMETRY == "custom_map_LD_385":
            answer = input("With this custom map, it is possible to now go through images of the guard ring on screen. Type in 'no' to skip, otherwise, press ENTER. \n")
            if answer != "no":
                guard_ring_scan_indices = sm.guard_ring_scan_steps
                for i in tqdm(guard_ring_scan_indices, total = len(guard_ring_scan_indices)):
                    try:
                        image_path = glob.glob(os.path.join(jpeg_output_dir, 're*step%s.jpeg' % int(i)))[0]
                    except:
                        image_path = glob.glob(os.path.join(jpeg_output_dir, 'init*step%s.jpeg' % int(i)))[0]
                    img = cv2.imread(image_path)
                    img = cv2.resize(img, (int(3840/4), int(2736/4)))
                    cv2.imshow("Guard ring", img)
                    clean = input("Press ENTER if image is clean. Type in 'END' to exit. If you type in anything else, the xy-stage will move into the correct position for cleaning. \n")
                    if clean.casefold() == 'END'.casefold():
                        cv2.destroyAllWindows()
                        break
                    elif clean == '':
                        cv2.destroyAllWindows()
                        continue
                    else:
                        _exists, current_pad_nr, _x, _y = sm.scan_step_coordinates(i)
                        if not _exists:
                            continue
                        sc.goto(_x, _y)
                        input("Press ENTER after done cleaning to continue. \n")
                        cv2.destroyAllWindows()
                        continue
                break
            elif answer == "no":
                print("Please manually clean the guard ring area now using the microscope and joystick.")
                break
            else:
                continue
        else:
            print("Please manually clean the guard ring area now using the microscope and joystick.")
            break

    print("Entering manual control. xy-stage will move to origin.")
    sc.goto(0, 0)
    sc.setManualMode()
    answer = input("Press ENTER to close.")
    lc.closeCommunication()
    sc.close()