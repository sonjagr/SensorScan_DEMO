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

from token import NAME
import numpy as np
import os, platform
import matplotlib.patches as patches
import cv2

from pypylon import pylon
from matplotlib import pyplot as plt
from abc import abstractmethod
from utility.decorators import ExecuteIfExists

class ImageEventBase(pylon.ImageEventHandler):
    def __init__(self):
        super().__init__()
        self.img = pylon.PylonImage()
        self.img_array = None # BAYER_RG

    @abstractmethod
    def action(self, camera, grabResult):
        raise NotImplementedError

    def get_image(self):
        return self.img

    def get_image_array(self):
        return self.img_array

    def OnImageGrabbed(self, camera, grabResult):
        # Image grabbed successfully?
        if grabResult.GrabSucceeded():
            self.img_array = grabResult.GetArray()  # format is BAYER_RG
            self.action(camera, grabResult)
        else:
            print("Error: ", grabResult.GetErrorCode(), grabResult.GetErrorDescription())

class ImageShowEvent(ImageEventBase):
    def __init__(self):
        super().__init__()
        self.matplot_fig = None
        self.matplot_ax = None
        self.converted_image = None
        self.matplot_image = None
        self.cmap = None
        self.matplot_fig, self.matplot_ax = None, None
        self.openFigure()

    def action(self, camera, grabResult):
        self.showImage()

    def setNone(self):
        self.matplot_fig, self.matplot_ax = None, None
        self.matplot_image = None

    def openFigure(self):
        if self.matplot_fig == None:
            self.matplot_fig, self.matplot_ax = plt.subplots(figsize=(10, 7))
            self.matplot_fig.canvas.mpl_connect('close_event', lambda x: self.setNone())
            self.matplot_fig.show()
            plt.pause(0.01)

    @ExecuteIfExists(ImageEventBase.get_image_array)
    def showImage(self):
        self.openFigure()
        if self.cmap == None:
            self.converted_image = cv2.cvtColor(self.img_array, cv2.COLOR_BAYER_RG2RGB)
        else:
            self.converted_image = self.img_array
        if self.matplot_image == None:
            self.matplot_image = self.matplot_ax.imshow(self.converted_image, cmap=self.cmap)
        else:
            self.matplot_image.set_data(self.converted_image)
        plt.pause(0.01)

    def updateFigure(self):
        plt.draw()
        plt.pause(0.01)

    def clear(self):
        if self.matplot_image:
            self.matplot_ax.clear()
        self.matplot_image = None

    def setColorMap(self, value):
        self.cmap = value
        self.clear()

    @ExecuteIfExists(ImageEventBase.get_image_array)
    def reloadImage():
        self.clear()
        self.showImage()

    def setColourMap(self, value):
        if value in ["viridis", "plasma", "inferno", "magma", "cividis"]:
            self.cmap = value
        else:
            print(value, "not a valid colour map")
            self.cmap = None
        self.reloadImage()

    def close(self):
        self.clear()
        plt.close(self.matplot_fig)


class ImageAnnotateEvent(ImageShowEvent):
    def __init__(self, _wx=160, _wy=144):
        super().__init__()
        self.WINDOWSIZE_X = _wx
        self.WINDOWSIZE_Y = _wy
        self.annotated_boxes = {}
        self.registerCallbacks()
        self.fpath = "pad"

    def registerCallbacks(self):
        self.matplot_fig.canvas.mpl_connect('button_press_event', lambda x: self.onclick(x))

    def addAnnotation(self, bID, x, y, AI=False):
        update_canvas = False
        if not bID in self.annotated_boxes:
            color = "r" if AI else "tab:orange"
            rp = patches.Rectangle((x, y), self.WINDOWSIZE_X, self.WINDOWSIZE_Y, linewidth=1, edgecolor=color,
                                   facecolor='none')
            self.annotated_boxes[bID] = (x, y, AI, rp)
            self.matplot_ax.add_patch(rp)
            update_canvas = True
        return update_canvas

    def removeAnnotation(self, bID):
        update_canvas = False
        if bID in self.annotated_boxes:
            self.annotated_boxes[bID][3].remove()
            del self.annotated_boxes[bID]
            update_canvas = True
        return update_canvas

    def removeAllAnnotations(self):
        all_annotations = [bID for bID in self.annotated_boxes]
        for bID in all_annotations:
            self.removeAnnotation(bID)

    def processAnnotations(self):
        NAnnotations = len([_k for _k in self.annotated_boxes])
        if bool(self.annotated_boxes):
            np.save("%s.npy" % self.fpath, self.img_array, allow_pickle=False)
            self.matplot_fig.savefig("%s.png" % self.fpath)
            self.dumpAnnotationsToFile("%s.txt" % self.fpath)
            self.removeAllAnnotations()
        return NAnnotations

    def dumpAnnotationsToFile(self, _fpath):
        _boxes = self.annotated_boxes
        _d = [(int(_boxes[_bID][2]), _boxes[_bID][0], _boxes[_bID][1], self.WINDOWSIZE_X, self.WINDOWSIZE_Y) for _bID in
              _boxes]
        _d = np.array(_d, dtype=int)
        np.savetxt(_fpath, _d, delimiter=",", fmt='%i', header="AI, x, y, dx, dy")

    def setFilePath(self, fpath):
        self.fpath = os.path.abspath(fpath)

    def onclick(self, event):
        try:
            x_bottom = int(event.xdata / self.WINDOWSIZE_X) * self.WINDOWSIZE_X
            y_bottom = int(event.ydata / self.WINDOWSIZE_Y) * self.WINDOWSIZE_Y
            box_ID = "%i-%i" % (x_bottom, y_bottom)
            update_canvas = False
            if event.dblclick:
                update_canvas = self.removeAnnotation(box_ID)
            else:
                update_canvas = self.addAnnotation(box_ID, x_bottom, y_bottom)
            if update_canvas:
                self.updateFigure()
        except:
            pass

from config import *
from tensorflow import keras
from utility.keras import setTrainable
class AIImageAnnotateEvent(ImageAnnotateEvent):
    def __init__(self, _wx=PICTURESIZE_X, _wy=PICTURESIZE_Y):
        super().__init__(_wx, _wy)
        self.classifier = None
        self.threshold = 0.5
        self.encoder = None
        self.decoder = None
        self.image_decoded = None

    # function for loading classifier
    def loadClassifierModel(self, fpath):
        self.classifier = keras.models.load_model(fpath)
        print("Loaded model saved in", fpath)
        print(self.classifier.summary())
        setTrainable(self.classifier, False)

    ## load autoencoder
    def loadAE(self, fpath):
        encoder_path = fpath + "_encoder"
        self.encoder = keras.models.load_model(encoder_path)
        print("Loaded encoder model saved in", encoder_path)
        print(self.encoder.summary())
        setTrainable(self.encoder, False)

        decoder_path = fpath + "_decoder"
        self.decoder = keras.models.load_model(decoder_path)
        print("Loaded decoder model saved in", decoder_path)
        print(self.decoder.summary())
        setTrainable(self.decoder, False)

    ## set threshold for classifier
    def setThreshold(self, _value=0.5):
        self.threshold = _value

    ## action function, take image and evaluate, OVERRIDES FUNCTION IN IMAGESHOWEVENT
    def action(self, camera, grabResult):
        super().action(camera, grabResult)
        self.evaluateConvertedImage()

    ## divide into patches
    def divide_img_blocks(self, img, n_blocks):
        horizontal = np.array_split(img, n_blocks[0], axis=1)
        splitted_img = [np.array_split(block, n_blocks[1], axis=2) for block in horizontal]
        return np.asarray(splitted_img, dtype=np.ndarray).reshape((-1, self.WINDOWSIZE_Y, self.WINDOWSIZE_X, 1)).astype(
            np.int16)

    def evaluateConvertedImage(self, _im=None):
        if _im == None:
            _image = np.expand_dims(self.img_array, axis=0)
        else:
            _image = np.expand_dims(_im, axis=0)

        # perform autoencoding and append to depth of the image
        z = self.encoder(_image)
        self.image_decoded = np.array(self.decoder(z)).astype(np.uint8)

        # compute difference
        _image = np.expand_dims(_image, axis=3).astype(np.int16)
        _image = np.abs(_image - self.image_decoded.astype(np.int16))

        # crop images and evaluate all individually
        cropped_boxes = REDUCED_DIMENSION
        _cropped_images = self.divide_img_blocks(_image, cropped_boxes)

        _classifier = np.squeeze(self.classifier(_cropped_images))
        annotations = np.transpose(np.array((_classifier > self.threshold)).nonzero())
        update_canvas = False
        for _annotation in annotations:
            _annotation_index = _annotation[0]
            x_bottom = (_annotation_index % cropped_boxes[1])
            y_bottom = (_annotation_index - x_bottom) / cropped_boxes[1]

            x_bottom *= self.WINDOWSIZE_X
            y_bottom *= self.WINDOWSIZE_Y
            box_ID = "%i-%i" % (x_bottom, y_bottom)
            update_canvas = self.addAnnotation(box_ID, x_bottom, y_bottom, AI=True) or update_canvas

        if update_canvas:
            self.updateFigure()

class ImageSaveEvent(ImageEventBase):
    def __init__(self, fpath="test.png"):
        super().__init__()
        self.img_path = None
        self.base_path = None
        self.extension = None
        if platform.system() == 'Windows':
            self.ipo = pylon.ImagePersistenceOptions()
        else:
            self.ipo = None
        self.setBasePath(fpath)
        self.setJPEGQuality()

    def setJPEGQuality(self, quality=50):
        quality = min(100, quality)
        quality = max(0, quality)
        if self.ipo:
            self.ipo.SetQuality(quality)

    def action(self, camera, grabResult):
        self.img.AttachGrabResultBuffer(grabResult)
        self.write_image_to_file(self.img_path)
        self.img.Release()

    def setBasePath(self, fpath):
        self.extension = fpath.split(".")[-1]
        self.img_path = os.path.abspath(fpath)
        self.base_path = self.img_path

    def setPathPostfix(self, postfix):
        self.img_path = self.base_path
        self.img_path = self.img_path.replace(".%s" % self.extension, "")
        self.img_path = self.img_path + "_" + postfix + "." + self.extension

    @ExecuteIfExists(ImageEventBase.get_image)
    def write_image_to_file(self, filename):
        if self.extension == "npy":
            np.save(filename, self.img_array, allow_pickle=False)
        elif self.extension == "jpeg" and self.ipo:
            self.img.Save(pylon.ImageFileFormat_Jpeg, filename, self.ipo)
        else:
            self.img.Save(pylon.ImageFileFormat_Png, filename)

class VerboseImageEvent(ImageEventBase):
    def __init__(self):
        super().__init__()
        self.counter = 0

    def action(self, camera, grabResult):
        print('Image nr. %i' % self.counter)
        print("OnImageGrabbed event for device ", camera.GetDeviceInfo().GetModelName())
        print("SizeX: ", grabResult.GetWidth())
        print("SizeY: ", grabResult.GetHeight())
        print()
        self.counter += 1