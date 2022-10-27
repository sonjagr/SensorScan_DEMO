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
#SOFTWARE

from classes.LEICAController import LEICAController
from classes.ImageHandler import *

if __name__ == "__main__":
    print
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
    print
    input("Press ENTER to take a photo:")
    print("Opening the image display...")
    ImageDisplay = ImageShowEvent()
    ImageDisplay.setColorMap(None)
    lc.setImageHandler(ImageDisplay)
    lc.takePhoto()
    input("Press ENTER to finish:")
    print
    print("Camera ready for usage!")
    print
