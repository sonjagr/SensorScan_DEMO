# Created by Thorben Quast on the 30th July 2021
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


import os
import argparse

from classes.StageController import StageController
from config import *

parser = argparse.ArgumentParser()
parser.add_argument("--COMPORT", type=int, help="COM port of the xy stage.", default=3, required=True)
args = parser.parse_args()

COMPORT = args.COMPORT

steps_x = [-8., 0, 8.]
steps_y =  [-8., 0, 8.]

from ctypes import WinDLL
dll_path = os.path.join(os.getcwd(), PRIOR_STAGE_DLL)

if __name__ == "__main__":
    if os.path.exists(dll_path):
        SDKPrior = WinDLL(dll_path)
    else:
        raise RuntimeError("DLL to control the xy stage could not be found.")

    print()
    print("Initialising the xy stage")
    sc = StageController(SDKPrior)
    sc.initialise_sequence()
    sc.APITest()
    sc.connectToPort(COMPORT)
    print("Serial number:", sc.getSerialNumber())
    print("Setting manual mode")
    sc.setManualMode()
    print("Going to (0, 0)")
    sc.goto(0, 0) 
    print("Setting true origin")
    sc.setTrueOrigin()
    print("Dummy scan...")
    sc.setScanMode()
    for x in range(0, 11, 2):
        sc.goto(x*10, 0)
    for y in range(0, 11, 2):
        sc.goto(100, y*10)
    sc.setManualMode()
    print("Back to (0, 0)")
    sc.goto(0, 0) 
    sc.close()
    print
    print("Stage ready for usage!")
    print