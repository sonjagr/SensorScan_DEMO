# Created by Thorben Quast
# Full sensor scanning, without AI-based feature detection.
# last modification: 02 August 2021

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
from ctypes import WinDLL
from utility.fs import create_dir
from config import *

from classes.StageController import StageController
from classes.LEICAController import LEICAController
from classes.ImageHandler import *
from classes.ScanMapping import ScanMapping

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--SensorGeometry", type=str, help="Sensor geometry (HPK_198ch_8inch, HPK_432ch_8inch).",
                    required=True)
parser.add_argument("--CampaignName", type=str,
                    help="Name of campaign.", required=True)
parser.add_argument("--DUTName", type=str,
                    help="Name of DUT.", required=True)
parser.add_argument("--DetailLevel", type=int, help="(option) Detail level of the xy-scan (0-coarsest, 2-finest).",
                    default=2, required=False)
parser.add_argument("--COMPORT", type=int, help="(option) COM port of the xy stage.", default=3, required=False)
args = parser.parse_args()

GEOMETRY = args.SensorGeometry
campaign = args.CampaignName
dut = args.DUTName

output_dir = create_dir(os.path.abspath(os.path.join("../outputs", campaign, dut)))

# COMPort for communcation to ProScan III xy stage controller
COMPORT = args.COMPORT

# indicate the path of the DLL to control the stage SW
dll_path = os.path.join(os.getcwd(), PRIOR_STAGE_DLL)

if __name__ == "__main__":
    if os.path.exists(dll_path):
        SDKPrior = WinDLL(dll_path)
    else:
        raise RuntimeError("DLL to control the xy stage could not be found.")

    print(" ----- 1 ----- ")
    print("Initialising the xy stage")
    sc = StageController(SDKPrior)
    sc.initialise_sequence()
    sc.APITest()
    sc.connectToPort(COMPORT)
    sc.getSerialNumber()
    sc.setManualMode()
    print("The xy-stage can be manually controlled now.")
    print()

    print(" ----- 2 ----- ")
    msg = "The xy-stage will perform a self-calibration next. \n"
    msg += "Please type 'no' to skip this step. \n"
    msg += "Otherwise, please remove any DUT from the stage. \n"
    answer = input(msg)
    if not answer == "no":
        sc.setTrueOrigin()

    print()
    print(" ----- 3 ----- ")
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

    print(" ----- 4 ----- ")
    msg = "We are about to place sensor. \n"
    msg += "The stage will now move to 0,0. \n"
    msg += "Please type 'no' to skip this step. \n"
    answer = input(msg)
    if not answer == "no":
        sc.goto(0, 0)
    print()

    print("Opening the image display...")
    ImageDisplay = ImageShowEvent()
    ImageDisplay.setColorMap(None)  # this accepts any of matplotlib's uniform sequential colormaps
    lc.setImageHandler(ImageDisplay)

    print()
    print(" ----- 5 ----- ")
    msg = "Please place the central pad underneath the microscope. \n"
    msg += "And please hit 'ENTER' afterwards. \n"
    input(msg)
    lc.takePhoto()
    while True:
        answer = input("Retake photo (yes/no)? \n")
        if answer == "yes":
            lc.takePhoto()
        elif answer == "no":
            break
        else:
            continue
    print()

    print(" ----- 6 ----- ")
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

    ImageDisplay.close()

    print(" ----- 7 ----- ")
    print("Loading the scan mapping")
    sm = ScanMapping(GEOMETRY)
    sm.loadGeoFile()
    sm.createPattern(detail_level=args.DetailLevel)
    sm.initialise_scan()
    sm.visualisePattern()
    print()

    ImageSave = ImageSaveEvent()
    lc.setImageHandler(ImageSave)
    ImageSave.setBasePath(os.path.join(output_dir, "initial_scan.jpeg"))
    ImageSave.setJPEGQuality(10)

    ImageSave2 = ImageSaveEvent()
    lc.appendImageHandler(ImageSave2)
    ImageSave2.setBasePath(os.path.join(output_dir, "initial_scan.npy"))

    print(" ----- 8 ----- ")
    input("Please hit 'ENTER' to initiate the wafer scan.")
    sc.setScanMode()
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

    print()
    print("Scan is finished.")
    print()
    sc.goto(0, 0)

    print(" ----- 9 ----- ")
    print("Revisiting suspicious areas")
    ImageSave.setBasePath(os.path.join(output_dir, "rescan.jpeg"))
    ImageSave2.setBasePath(os.path.join(output_dir, "rescan.npy"))

    clean_event = False
    msg = "Type in step index of sensor area to re-visit. \n"
    msg += "Type 'END' to stop. \n"
    while True:
        answer = input(msg)
        if clean_event:
            lc.takePhoto()
            clean_event = False
        if answer == "END":
            break
        else:
            try:
                step = int(answer)
            except:
                print("Input must be an integer.")
                continue
            _exists, current_pad_nr, _x, _y = sm.scan_step_coordinates(step)
            if not _exists:
                continue
            ImageSave.setPathPostfix("pad%i_step%i" % (current_pad_nr, step))
            ImageSave2.setPathPostfix("pad%i_step%i" % (current_pad_nr, step))
            sc.goto(_x, _y)
            sm.setNAnnotations(step, 1)
            clean_event = True

    print()
    print(" ----- 10 ----- ")
    print("End of the scanning program.")
    print("Please manually clean the guard ring area now using the microscope and joystick.")
    sc.goto(0, 0)
    sc.setManualMode()
    # closing everything
    lc.closeCommunication()
    sc.close()
