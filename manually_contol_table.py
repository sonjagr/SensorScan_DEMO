# original author: Thorben Quast
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
from ctypes import WinDLL
from classes.StageController import StageController
from classes.ImageHandler import *

# COMPort for communication to ProScan III xy stage controller
COMPORT = 3

# indicate the path of the DLL to control the stage SW
dll_path = os.path.join(os.getcwd(), PRIOR_STAGE_DLL)

if __name__ == "__main__":
    if os.path.exists(dll_path):
        SDKPrior = WinDLL(dll_path)
    else:
        raise RuntimeError("DLL to control the xy stage could not be found.")

    print("Initializing the xy stage for manual control.")
    sc = StageController(SDKPrior)
    sc.initialise_sequence()
    sc.APITest()
    sc.connectToPort(COMPORT)
    sc.getSerialNumber()
    while True:
        answer = input("Do you want the xy-table to self-calibrate (yes/no)? \n")
        if answer == 'yes':
            sc.setTrueOrigin()
            print("Going to origin.")
            sc.goto(0, 0)
            sc.setManualMode()
            print("The xy-stage is in origin and can be manually controlled now.")
            break
        elif answer == 'no':
            sc.setManualMode()
            print("The xy-stage can be manually controlled now.")
            break
        continue
    print()
