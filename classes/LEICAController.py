# Created by Thorben Quast, August 2021
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

import time

from pypylon import pylon
from utility.decorators import ExecuteIfExists
from config import *

class CameraEvent(pylon.ConfigurationEventHandler):
    def __init__(self, _maxNumBuffer, _exposureTime):
        super().__init__()
        self._MaxNumBuffer = _maxNumBuffer  # buffer holds only one image
        self._ExposureTime = _exposureTime  # microseconds

    def OnAttached(self, camera):
        print(f'Attached: {camera.GetDeviceInfo()}')

    def OnOpened(self, camera):
        # default camera settings
        camera.MaxNumBuffer = self._MaxNumBuffer
        camera.Gain = camera.Gain.Max
        camera.Width = PICTURESIZE_X
        camera.Height = PICTURESIZE_Y
        camera.ExposureTime = self._ExposureTime * 1E3
        camera.PixelFormat = "BayerBG8"

    def OnGrabStarted(self, camera):
        time.sleep(self._ExposureTime / 1e6)

    def OnOpen(self, camera):
        pass

    def OnDestroy(self, camera):
        pass

    def OnDestroyed(self, camera):
        pass

    def OnClosed(self, camera):
        pass

    def OnDetach(self, camera):
        pass


class CameraController:
    def __init__(self):
        self.camera = None

    def getCamera(self):
        return self.camera


class LEICAController(CameraController):
    def __init__(self):
        super().__init__()
        self.tlf = pylon.TlFactory.GetInstance()
        self.devices = self.tlf.EnumerateDevices()

        self.camera_event = None
        self.image_handler = None

    def showAllConnectedDevices(self):
        for i, device in enumerate(self.devices):
            print(i, ":")
            print("Friendly name: ", device.GetFriendlyName())
            print("Full name: ", device.GetFullName())
            print("Serial number: ", device.GetSerialNumber())
            print

    def connectCamera(self, MaxNumBuffer=1, ExposureTime=10):
        if len(self.devices) > 1:
            print("More than one device connected. Not sure to which to connect...")
            return
        self.camera = pylon.InstantCamera()
        self.camera_event = CameraEvent(MaxNumBuffer, ExposureTime)
        self.camera.RegisterConfiguration(self.camera_event, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_Delete)
        self.camera.Attach(self.tlf.CreateFirstDevice())

    @ExecuteIfExists(CameraController.getCamera)
    def getConnectedModelName(self):
        return self.camera.GetDeviceInfo().GetModelName()

    @ExecuteIfExists(CameraController.getCamera)
    def startCommunication(self):
        self.camera.Open()

    @ExecuteIfExists(CameraController.getCamera)
    def setImageHandler(self, _ih):
        self.image_handler = _ih
        self.camera.RegisterImageEventHandler(self.image_handler, pylon.RegistrationMode_ReplaceAll,
                                              pylon.Cleanup_Delete)

    @ExecuteIfExists(CameraController.getCamera)
    def appendImageHandler(self, _ih):
        self.image_handler = _ih
        self.camera.RegisterImageEventHandler(self.image_handler, pylon.RegistrationMode_Append, pylon.Cleanup_Delete)

    @ExecuteIfExists(CameraController.getCamera)
    def closeCommunication(self):
        self.camera.Close()

    @ExecuteIfExists(CameraController.getCamera)
    def deleteCamera(self):
        self.camera = None

    @ExecuteIfExists(CameraController.getCamera)
    def setExposureTime(self, val):
        self.camera.ExposureTime = val * 1E3

    @ExecuteIfExists(CameraController.getCamera)
    def setMaxNumBuffer(self, val):
        self.camera.MaxNumBuffer = val

    @ExecuteIfExists(CameraController.getCamera)
    def getExposureTime(self):
        return self.camera.ExposureTime.GetValue() / 1E3

    @ExecuteIfExists(CameraController.getCamera)
    def getMaxNumBuffer(self):
        return self.camera.MaxNumBuffer.GetValue()

    @ExecuteIfExists(CameraController.getCamera)
    def startGrabbing(self, N=1):
        self.camera.StartGrabbing(N)

    @ExecuteIfExists(CameraController.getCamera)
    def stopGrabbing(self):
        self.camera.StopGrabbing()

    @ExecuteIfExists(CameraController.getCamera)
    def retrieveImage(self, enforced=False):
        while True:
            result = self.camera.RetrieveResult(RDOUTTIMEOUT, pylon.TimeoutHandling_Return)
            if result == None:
                if enforced:
                    continue
                else:
                    print("No image could be retrieved.")
                    break
            else:
                break

    @ExecuteIfExists(CameraController.getCamera)
    def takePhoto(self):
        self.startGrabbing(1)
        self.retrieveImage()
        self.stopGrabbing()
