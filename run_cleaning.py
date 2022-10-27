# Created by Sonja Grönroos in August 2022
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
import argparse
import msvcrt, time

from ctypes import WinDLL
from config import *
from matplotlib import pyplot as plt
from utility.fs import create_dir
from classes.StageController import StageController
from classes.LEICAController import LEICAController
from classes.ImageHandler import ImageShowEvent, VerboseImageEvent, ImageSaveEvent
from classes.ScanMapping import ScanMapping

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

campaign = args.CampaignName
dut = args.DUTName
output_dir = create_dir(os.path.abspath(os.path.join("outputs", campaign, dut)))
GEOMETRY = args.SensorGeometry
DL = args.DetailLevel
if "custom" in GEOMETRY:
    DL = 0
# COMPort for communication to ProScan III xy stage controller
COMPORT = args.COMPORT
# indicate the path of the DLL to control the stage SW
dll_path = os.path.join(os.getcwd(), PRIOR_STAGE_DLL)

print()
print("--- STARTING CLEANING PROGRAM FOR DUT %s IN CAMPAIGN %s ---" % (dut, campaign))
print("This program is to be run after the pre-selection (PS) analysis has finished. \n ")

if os.path.exists(dll_path):
    SDKPrior = WinDLL(dll_path)
else:
    raise RuntimeError("DLL to control the xy stage could not be found.")

print('Setting up the camera, stage controller and the scan mapping...')
lc = LEICAController()
print("The connected cameras are:")
lc.showAllConnectedDevices()
print("Connecting...")
lc.connectCamera()
print(lc.getConnectedModelName())
lc.setImageHandler(VerboseImageEvent())
print("Starting the communication...")
lc.startCommunication()

sc = StageController(SDKPrior)
sc.initialise_sequence()
sc.APITest()
sc.connectToPort(COMPORT)
sc.getSerialNumber()
#sc.setTrueOrigin()
sc.goto(0, 0)

sm = ScanMapping(GEOMETRY)
sm.loadGeoFile()
sm.createPattern(detail_level=DL)
sm.initialise_scan()
sm.visualisePattern()
print()

jpeg_output_dir = create_dir(os.path.abspath(os.path.join(output_dir, 'jpegs')))

print(" ----- Visual inspection ----- ")
print("Revisiting suspicious areas \n")

try:
    f = open(os.path.join(output_dir, "annotation_indices.txt"), 'r')
    last_line = f.readline()
    f.close()
    AI_scan_steps = list(map(int, last_line.split(' ')[:-1]))
    print("%s images have been pre-selected. \n" % len(AI_scan_steps))
    for i in AI_scan_steps:
        sm.setNAnnotations(i, 1)
except:
    print("Pre-selected scan indexes do not exist/could not be read. \n")
    AI_scan_steps = []

plt.close(2)
sc.setScanMode()
sc.goto(0, 0)
while True:
    print("Type in step index of sensor area to re-visit.")
    if len(AI_scan_steps) > 0 and AI_scan_steps is not None:
        print("  Pre-selection suggests you visit one of these step indexes: %s" % AI_scan_steps)
    elif len(AI_scan_steps) == 0:
        print("  No pre-selected suggestions. \n")
    answer = input("Type 'END' to stop. \n")
    if answer == "END":
        break
    else:
        try:
            step = int(answer)
        except:
            print("Input must be an integer.")
            continue
        if len(AI_scan_steps) > 0 and step in AI_scan_steps:
            AI_scan_steps.remove(step)
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
            answer = input("Take video at scan index %s to show cleaning (yes/no)?\n" % step)
            if answer.casefold() == "yes".casefold():
                i = 0
                while True:
                    lc.takePhoto()
                    if i == 0:
                        print('Taking photos. Press any key to quit cleaning: a rescan .jpeg will be taken of the cleaned area and saved. \n', flush=True)
                    if msvcrt.kbhit():
                        break
                    time.sleep(0.01)
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

print()
print(" ----- 11: End. ----- ")
print("End of the scanning program.")
print("Please manually clean the guard ring area now using the microscope and joystick.")
sc.goto(0, 0)
sc.setManualMode()

lc.closeCommunication()
sc.close()