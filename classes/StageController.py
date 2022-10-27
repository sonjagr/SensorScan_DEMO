# Created by Thorben Quast, August 2021
# Last modified: February 2022
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

import sys, time
from ctypes import create_string_buffer

rx = create_string_buffer(1000)

# default length unit is mm
LENGTHSCALE = 1e3  # mm to mum


class StageController:
    def __init__(self, _SDKPrior, verbose=False):
        self.SDKPrior = _SDKPrior
        self.initialised = False
        self.sessionID = None
        self.verbose = verbose

    '''
        DLL-API functions.
    '''

    def initialise(self):
        # Initialise the stage controller SW
        ret = self.SDKPrior.PriorScientificSDK_Initialise()
        if ret:
            print(f"Error initialising {ret}")
            sys.exit()
        else:
            print(f"Ok initialising {ret}")

    def getVersion(self):
        ret = self.SDKPrior.PriorScientificSDK_Version(rx)
        version = rx.value.decode()
        print(f"DLL version of the xy-stage is {version}")
        return version

    def openSession(self):
        sessionID = self.SDKPrior.PriorScientificSDK_OpenNewSession()
        if sessionID < 0:
            print(f"Error getting sessionID {ret}")
        else:
            if self.verbose:
                print(f"SessionID = {sessionID}")
            self.sessionID = sessionID
        return sessionID

    def closeSession(self):
        self.SDKPrior.PriorScientificSDK_CloseSession(self.sessionID)
        self.sessionID = None

    def APITest(self):
        # API Test
        ret = self.SDKPrior.PriorScientificSDK_cmd(
            self.sessionID, create_string_buffer(b"dll.apitest 33 goodresponse"), rx
        )
        print(f"api response {ret}, rx = {rx.value.decode()}")

    # command function to send them to the ProScan III controller --> stage
    def cmd(self, msg):
        ret = self.SDKPrior.PriorScientificSDK_cmd(
            self.sessionID, create_string_buffer(msg.encode()), rx
        )
        if ret:
            print(f"Stage API error {ret}")
        else:
            if self.verbose:
                print(f"OK {rx.value.decode()}")

        return ret, rx.value.decode()

    '''
        Elementary xy stage control functions.
    '''

    def connectToPort(self, com):
        if self.verbose:
            print("Connecting to COM port %i..." % com)
        self.cmd("controller.connect %i" % com)

    def getSerialNumber(self):
        _, sn = self.cmd("controller.serialnumber.get")
        if self.verbose:
            print(f"Controller serial number is {sn}.")
        return sn

    def busy(self):
        # see busy status
        _c_, busy = self.cmd("controller.stage.busy.get")
        busy = bool(int(busy))
        return busy

    def gotoposition(self, x, y):
        self.cmd("controller.stage.goto-position " + str(x * LENGTHSCALE) + " " + str(y * LENGTHSCALE))
        if self.verbose:
            print(f"Going to position {x},{y} [mm]")

    def move(self, vx, vy):
        self.cmd("controller.stage.move-at-velocity " + str(vx * LENGTHSCALE) + " " + str(vy * LENGTHSCALE))
        if self.verbose:
            print(f"Moving stage by {vx},{vy} [mm/s]")

    def stop(self, abrupt=False):
        if abrupt:
            self.cmd("controller.stop.abruptly")
        else:
            self.cmd("controller.stop.smoothly")
        if self.verbose:
            print("Stopped movement of the stage")

    def getPosition(self):
        _, xy = self.cmd("controller.stage.position.get")
        x, y = tuple(xy.split(","))
        x = float(x) / LENGTHSCALE
        y = float(y) / LENGTHSCALE
        if self.verbose:
            print(f"Stage position is {x},{y} [mm]")
        return x, y

    def setPosition(self, x, y):
        self.cmd("controller.stage.position.set " + str(x * LENGTHSCALE) + " " + str(y * LENGTHSCALE))
        if self.verbose:
            print(f"Set current position to {x},{y} [mm]")

    def getVelocity(self):
        _, v = self.cmd("controller.stage.speed.get")
        v = float(v) / LENGTHSCALE
        if self.verbose:
            print(f"Moving velocity is {v} [mm/s]")
        return v

    def setVelocity(self, v):
        self.cmd("controller.stage.speed.set " + str(v * LENGTHSCALE))
        if self.verbose:
            print(f"Set moving velocity to {v} [mm/s]")

    def getAcceleration(self):
        _, a = self.cmd("controller.stage.acc.get")
        a = float(a) / LENGTHSCALE
        if self.verbose:
            print(f"Moving acceleration is {a} [mm/s]")
        return a

    def setAcceleration(self, a):
        self.cmd("controller.stage.acc.set " + str(a * LENGTHSCALE))
        if self.verbose:
            print(f"Set moving acceleration to {a} [mm/s]")

    def setHostDirection(self, ox=1, oy=1):
        self.cmd("controller.stage.hostdirection.set " + str(ox) + " " + str(oy))
        if self.verbose:
            print(f"Host direction is set to {ox},{oy}")

    def enableJoystick(self, enable=True):
        _cmd = "controller.stage.joyxyz.on"
        if not enable:
            _cmd = _cmd.replace(".on", ".off")
        self.cmd(_cmd)

    def disconnect(self):
        if self.verbose:
            print("Disconnecting the controller")
        self.cmd("controller.disconnect")

    '''
        Composite, more sophisticated user functions.
    '''

    # this sets up the SDK SW and opens a new session
    def initialise_sequence(self):
        if self.initialised:
            return
        self.initialise()
        self.getVersion()
        self.openSession()
        self.initialise = True

    def setScanMode(self):
        self.enableJoystick(False)
        self.setVelocity(40.0)  # 40 mm/s
        self.setAcceleration(50.0)  # 50 mm/s2
        self.setHostDirection(-1, 1)

    def setManualMode(self):
        self.enableJoystick(True)
        self.setVelocity(20.0)  # 20 mm/s
        self.setAcceleration(150.0)  # 150 mm/s2
        self.setHostDirection(1, 1)

    def goto(self, x, y):
        self.gotoposition(x, y)
        # see busy status
        busy = self.busy()
        while busy:
            time.sleep(0.1)
            busy = self.busy()

    def gotolimit(self, direction):
        v = self.getVelocity()
        if direction < 0:
            v = -v
        self.move(v, v)
        # see busy status
        busy = self.busy()
        while busy:
            time.sleep(0.1)
            busy = self.busy()

    def setTrueOrigin(self):
        v_set = self.getVelocity()
        a_set = self.getAcceleration()
        v_here = 25
        a_here = 100
        self.setVelocity(v_here)
        self.setAcceleration(a_here)
        # go to one side
        self.gotolimit(1)
        xup, yup = self.getPosition()
        # go to the other side
        self.gotolimit(-1)
        xdown, ydown = self.getPosition()

        # calculate and move to center
        xcenter = (xup + xdown) / 2.
        ycenter = (yup + ydown) / 2.
        self.goto(xcenter, ycenter)
        # redefine settings
        self.setPosition(0, 0)
        self.setVelocity(v_set)
        self.setAcceleration(a_set)

    def close(self):
        self.disconnect()
        self.closeSession()


if __name__ == "__main__":
    # this sequence re-initialises the xy stage, e.g. when manual operation is turned off after failure / interrupt of the scanning program
    COMPORT = 4
    from config import *
    import os
    from ctypes import WinDLL

    # indicate the path of the DLL to control the stage SW
    dll_path = os.path.join(os.getcwd(), PRIOR_STAGE_DLL)
    if os.path.exists(dll_path):
        SDKPrior = WinDLL(dll_path)
    else:
        raise RuntimeError("DLL to control the xy stage could not be found.")

    print("Initialising the xy stage")
    sc = StageController(SDKPrior)
    sc.initialise_sequence()
    sc.APITest()
    sc.connectToPort(COMPORT)
    sc.getSerialNumber()
    sc.setManualMode()
    print("The xy-stage can be manually controlled now.")
    print()
