# Created by Thorben Quast
# Scan demonstration code incl. AI-based feature detection and user interaction.
# Concept described in section 2.2 in this white paper: https://arxiv.org/abs/2203.08969
# Modified by Sonja Grönroos in August 2022

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

import numpy as np
import os, platform
import cv2
import glob
import pandas as pd
import matplotlib.patches as patches
import random
random.seed(42)

from beautifultable import BeautifulTable
from tqdm import tqdm
from pypylon import pylon
from utility.fs import box_index_to_coords, box_index_to_coords_2
from matplotlib import pyplot as plt
from abc import abstractmethod
from utility.decorators import ExecuteIfExists

pd.options.mode.chained_assignment = None

class ImageEventBase(pylon.ImageEventHandler):
    def __init__(self):
        super().__init__()
        self.img = pylon.PylonImage()
        self.img_array = None

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
            self.img_array = grabResult.GetArray()              #format is BAYER_RG
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

    def set_img(self, array):
        self.img_array = array

    def action(self):
        self.showImage()

    def setNone(self):
        self.matplot_fig, self.matplot_ax = None, None
        self.matplot_image = None

    def openFigure(self):
        if self.matplot_fig == None:
            self.matplot_fig, self.matplot_ax = plt.subplots(figsize=(10, 7))
            self.matplot_fig.canvas.mpl_connect('close_event', lambda x: self.setNone())
            plt.pause(0.01)

    def openFigure_noshow(self):
        if self.matplot_fig == None:
            self.matplot_fig, self.matplot_ax = plt.subplots(figsize=(10, 7))
            self.matplot_fig.canvas.mpl_connect('close_event', lambda x: self.setNone())

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

    def set_label(self, string):
        plt.title(string)

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
    def reloadImage(self):
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
            np.save(filename, self.img_array)
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

from config import *
class ImageAnnotateEvent(ImageShowEvent):
    def __init__(self, _wx=PATCHSIZE, _wy=PATCHSIZE):
        super().__init__()
        self.WINDOWSIZE_X = _wx
        self.WINDOWSIZE_Y = _wy
        self.annotated_boxes = {}
        self.registerCallbacks()
        self.filename = ''
        self.grid = 1
        self.grid_status = 1
        self.fpath = None
        self.DBLocation = os.path.join(os.getcwd(), 'db_test')
        self.Campaign = 'None'
        self.DUT = 'None'
        self.databaseName = "annotation_DB" ## can be absolute, this does not change
        #self.databaseName = "annotation_DB"
        self.AI = True
        self.matplot_image

    def set_DUT(self, dut):
        self.DUT = dut

    def set_Campaign(self, c):
        self.Campaign = c

    def set_filename(self, filename):
        self.filename = filename

    def set_img(self, matplot_fig):
        super().set_img(matplot_fig)

    def registerCallbacks(self):
        self.matplot_fig.canvas.mpl_connect('button_press_event', lambda x: self.onclick(x))

    ## adds annotation by using the box coordinated
    def addAnnotation(self, bID, x, y):
        update_canvas = False
        if not bID in self.annotated_boxes:
            color = "r" if self.AI else "tab:orange"
            rp = patches.Rectangle((x, y), self.WINDOWSIZE_X, self.WINDOWSIZE_Y, linewidth=1, edgecolor=color,facecolor='none')
            self.annotated_boxes[bID] = (x, y, self.AI, rp)
            self.matplot_ax.add_patch(rp)
            update_canvas = True
        return update_canvas

    ## adds annotation by using the box index and if it is predicted by AI
    def addAnnotation_AI(self, p, grid):
        update_canvas = False
        if grid == 1:
            x_p, y_p = box_index_to_coords(p)
        elif grid == 2:
            x_p, y_p = box_index_to_coords_2(p)
        bID = "%i-%i" % (x_p, y_p)
        if not bID in self.annotated_boxes and self.AI:
            color = "r" if self.AI else "tab:orange"
            rp = patches.Rectangle((x_p, y_p), self.WINDOWSIZE_X, self.WINDOWSIZE_Y, linewidth=1, edgecolor=color,facecolor='none')
            self.annotated_boxes[bID] = (x_p, y_p, self.AI, rp)
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
            np.save("%s.npy" % self.fpath, self.img_array, allow_pickle=True)
            self.matplot_fig.savefig("%s.png" % self.fpath)
            _boxes = self.annotated_boxes
            if self.save_to_database:
                self.saveAnnotationsToDatabase(os.path.join(self.DBLocation, self.databaseName), _boxes, self.DUT, self.Campaign)
            self.removeAllAnnotations()
        return NAnnotations

    def saveAnnotationsToDatabase(self, _fpath, _boxes, _dut, _campaign):
        is_database = os.path.exists(_fpath)
        filename = self.filename
        _d = [(filename, "%s-%s" % (_boxes[_bID][0], _boxes[_bID][1]), int(_boxes[_bID][2]), _boxes[_bID][0],
               _boxes[_bID][1]) for _bID in _boxes]
        _d = np.array(_d)
        new_database = pd.DataFrame(data=_d, columns=["FileName", "bID", "AI", "x", "y"])
        new_database['AIDate'] = pd.to_datetime('today').date()
        new_database['Campaign'] = _campaign
        new_database['DUT'] = _dut
        new_database = new_database.set_index(['Campaign', 'DUT', 'FileName', 'bID'])
        if is_database:
            old_database = pd.read_pickle(_fpath)
            database = pd.concat([old_database,new_database])
        else:
            database = new_database
        database = database[~database.index.duplicated(keep='last')]
        database.to_pickle(_fpath)

    def saveAnnotationsToValidationDatabase(self, _fpath, _fname, _dut, _campaign, normal):
        is_database = os.path.exists(_fpath)
        _boxes = self.annotated_boxes
        filename = self.filename
        if len(_boxes) > 0:
            _d = [(filename, "%s-%s" % (_boxes[_bID][0], _boxes[_bID][1]), int(_boxes[_bID][2]), _boxes[_bID][0], _boxes[_bID][1]) for _bID in _boxes]
        if len(_boxes) == 0:
            _d = [(filename, np.nan, 0, np.nan, np.nan)]
        _d = np.array(_d)
        new_database = pd.DataFrame(data=_d, columns=["FileName", "bID", "AI", "x", "y"])
        new_database['ValDate'] = pd.to_datetime('today').date()
        new_database['Campaign'] = _campaign
        new_database['DUT'] = _dut
        new_database['Normal'] = normal
        new_database = new_database.set_index(['Campaign', 'DUT', 'FileName', 'bID'])
        if is_database:
            old_database = pd.read_pickle(_fpath)
            database = pd.concat([old_database, new_database])
        else:
            database = new_database
        database = database[~database.index.duplicated(keep='last')]
        database.to_pickle(_fpath)

    def removeOldAnnotationsFromValidationDatabase(self, _fpath, _fname, _dut, _campaign):
        is_database = os.path.exists(_fpath)
        if is_database:
            database = pd.read_pickle(_fpath).reset_index()
            database = database.drop(database[(database['Campaign'] == _campaign) & (database['DUT'] == _dut) & (database['FileName'] == _fname)].index)
            database = database.set_index(['Campaign', 'DUT', 'FileName', 'bID'])
        else:
            print('ERROR in deletion')
        database.to_pickle(_fpath)

    def setFilePath(self, fpath):
        self.fpath = os.path.abspath(fpath)

    def setAI(self, AI):
        self.AI = AI

    def onclick(self, event):
        try:
            if self.grid_status == 1:
                x_bottom = int(event.xdata / self.WINDOWSIZE_X) * self.WINDOWSIZE_X
                y_bottom = int(event.ydata / self.WINDOWSIZE_Y) * self.WINDOWSIZE_Y
            elif self.grid_status == 2:
                x_bottom = int(event.xdata / self.WINDOWSIZE_X) * self.WINDOWSIZE_X + 80
                y_bottom = int(event.ydata / self.WINDOWSIZE_Y) * self.WINDOWSIZE_Y + 80
                x_bottom = x_bottom if x_bottom <= 23*160 - 80 else 23*160 - 80
                y_bottom = y_bottom if y_bottom <= 16 * 160 - 80 else 16 * 160 - 80
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

class ImageValidateEvent(ImageAnnotateEvent):
    def __init__(self, _wx=PATCHSIZE, _wy=PATCHSIZE):
        super().__init__(_wx, _wy)
        self.fraction = 1
        self.bIDlist = []
        self.filesToValidate = []
        self.anomalous_scan_steps_npy = []
        self.validatedDatabaseName = "validation_DB"
        #self.validatedDatabaseName = "validation_DB_testing"
        self.validation = True
        self.path_to_file = None

    def set_frac(self, fraction):
        self.fraction = fraction

    def get_fileNames(self, _fpath, _databasename, _dut, _campaign, normal = None):
        path = os.path.join(_fpath, _databasename)
        is_database = os.path.exists(path)
        if not is_database:
            print("  Database does not exist (yet)")
        else:
            database = pd.read_pickle(path)
            try:
                database = database.loc[[_campaign]]
            except:
                return []
            database = database.reset_index()
            if _dut in database['DUT'].tolist():
                database = database[database.Campaign == _campaign]
                database = database[database.DUT == _dut]
                if normal is not None:
                    database = database[database.Normal == normal]
                files = list(dict.fromkeys(database['FileName'].tolist()))
                return files
            else:
                return []

    def get_annotations(self, _fpath, databasename, _fileName, _dut,  _campaign):
        path = os.path.join(_fpath, databasename)
        is_database = os.path.exists(path)
        if not is_database:
            print("  ERROR: Database does not exist")
        else:
            database = pd.read_pickle(path)
            database = database.loc[[_campaign]].loc[(slice(None), _dut), :]
            database = database.reset_index()
            database = database[database.FileName == _fileName]
            self.bIDlist = database['bID'].tolist()
            return database['bID'].tolist()

    def processValidationAnnotations(self, overwrite = False, normal = False):
        NAnnotations = len([_k for _k in self.annotated_boxes])
        if overwrite == True:
            self.removeOldAnnotationsFromValidationDatabase(os.path.join(self.DBLocation, self.validatedDatabaseName), self.filename.split('.')[0], self.DUT, self.Campaign)
        self.saveAnnotationsToValidationDatabase(os.path.join(self.DBLocation, self.validatedDatabaseName), self.filename.split('.')[0], self.DUT, self.Campaign, normal)
        self.removeAllAnnotations()
        return NAnnotations

    def validation_action(self, type, overwrite = False):
        typelist = []
        normal_to_validate = glob.glob(os.path.join(self.fpath, "init*.npy"))
        normal_to_validate = [item.split("\\")[-1].split(".")[0] for item in normal_to_validate]
        anom_to_validate = self.get_fileNames(self.DBLocation, self.databaseName, self.DUT, self.Campaign, None)
        _validated = self.get_fileNames(self.DBLocation, self.validatedDatabaseName, self.DUT, self.Campaign, None)
        if type == "normal":
            _to_validate = normal_to_validate
            if _validated is not None and overwrite == False:
                _to_validate = list(set(_to_validate) - set(_validated))
            typelist = np.full(len(_to_validate), 'normal')
        elif type == "anomalous":
            _to_validate = anom_to_validate
            if _validated is not None and overwrite == False:
                _to_validate = list(set(_to_validate) - set(_validated))
            typelist = np.full(len(_to_validate), 'anomalous')
        elif type == 'all':
            if _validated is not None and overwrite == False:
                normal_to_validate = list(set(normal_to_validate) - set(_validated))
                anom_to_validate = list(set(anom_to_validate) - set(_validated))
            _to_validate = np.append(normal_to_validate, anom_to_validate)
            typelist = np.append(np.full(len(normal_to_validate), 'normal'),  np.full(len(anom_to_validate), 'anomalous'))
            c = list(zip(_to_validate, typelist))
            random.shuffle(c)
            _to_validate, typelist = zip(*c)
        if len(_to_validate) > 0:
            print('You will validate %s files:' % len(_to_validate))
            for _f,_t in tqdm(zip(_to_validate, typelist), total = len(_to_validate)):
                scan_index = _f.split("step", 1)[1]
                self.clear()
                self.set_filename(_f)
                self.path_to_file = os.path.join(self.fpath, self.filename + '.npy')
                try:
                    self.set_img(np.load(self.path_to_file))
                except:
                    print(self.path_to_file, " does not exist, continuing")
                    continue
                if _t == 'normal':
                    normal = True
                    self.bIDlist = []
                    self.grid_status = 1
                elif _t == 'anomalous':
                    normal = False
                    self.bIDlist = self.get_annotations(self.DBLocation, self.databaseName, self.filename, self.DUT, self.Campaign)
                    first_x, _ = self.bIDlist[0].split('-')
                    if int(first_x) == 0 or int(first_x) % 160 == 0:
                        self.grid_status = 1
                    else:
                        self.grid_status = 2
                self.showImage()
                self.registerCallbacks()
                self.evaluateAnnotation()
                print("   Currently evaluating file: ", _f)
                cont = input('  Press enter for next image, write "END" to exit validation. \n')
                N_ann = self.processValidationAnnotations(overwrite, normal)
                if N_ann == 0 and self.anomalous_scan_steps_npy is not None:
                    print("  Validation result: normal image")
                    a = np.where(self.anomalous_scan_steps_npy == int(scan_index))
                    self.anomalous_scan_steps_npy = np.delete(self.anomalous_scan_steps_npy, a)
                    self.anomalous_scan_steps_npy = np.unique(self.anomalous_scan_steps_npy)
                elif N_ann > 0 and self.anomalous_scan_steps_npy is not None:
                    print("  Validation result: anomalous image")
                    self.anomalous_scan_steps_npy = np.append(self.anomalous_scan_steps_npy, int(scan_index))
                    self.anomalous_scan_steps_npy = np.unique(self.anomalous_scan_steps_npy)
                if cont.casefold() == 'END'.casefold():
                    break
        else:
            print("No files to validate.")

    def evaluateAnnotation(self, _im=None):
        if _im == None:
            _image = np.load(self.path_to_file)
            _image = np.expand_dims(_image, axis=0)
        else:
            _im = np.load(_im)
            _image = np.expand_dims(_im, axis=0)

        update_canvas = False
        self.setAI(True)
        for _bID in self.bIDlist:
            x_p, y_p = _bID.split('-')
            x_p = int(x_p)
            y_p = int(y_p)
            self.addAnnotation(_bID, x_p, y_p)
            self.updateFigure()
        self.setAI(False)

        if update_canvas:
            self.updateFigure()

    def printCM(self, tp, tn, fp, fn):
        table = BeautifulTable()
        table.columns.header = ["", "Predicted normal ", "Predicted anomalous"]
        table.rows.append(['True normal', tn, fp])
        table.rows.append(['True anomalous', fn, tp])
        print(table)

    def createValidationSummary(self):
        _normal_validated_files = self.get_fileNames(self.DBLocation, self.validatedDatabaseName, self.DUT, self.Campaign, True)
        _anomalous_validated_files = self.get_fileNames(self.DBLocation, self.validatedDatabaseName, self.DUT, self.Campaign, False)
        _matches, _fps, _fns, nbr_anomalous_human_clean, nbr_normal_human_clean = 0,0,0,0,0
        _normal_validated = len(_normal_validated_files)
        _anomalous_validated = len(_anomalous_validated_files)

        for _f in _anomalous_validated_files:
            _AI_bID = self.get_annotations(self.DBLocation, self.databaseName, _f, self.DUT, self.Campaign)
            _human_bID = self.get_annotations(self.DBLocation, self.validatedDatabaseName, _f, self.DUT, self.Campaign)
            if 'nan' in _human_bID:
                _human_bID = _human_bID.remove('nan')
            if _human_bID is not None:
                match = list(set(_AI_bID).intersection(_human_bID))
                not_in_AI = list(set(_human_bID) - set(_AI_bID))
                not_in_HU = list(set(_AI_bID) - set(_human_bID))
            else:
                nbr_anomalous_human_clean = nbr_anomalous_human_clean + 1
                match = []
                not_in_AI = []
                not_in_HU = _AI_bID
            _matches += len(match)
            _fps += len(not_in_HU)
            _fns += len(not_in_AI)

        for _f in _normal_validated_files:
            _AI_bID = self.get_annotations(self.DBLocation, self.databaseName, _f, self.DUT, self.Campaign)
            _human_bID = self.get_annotations(self.DBLocation, self.validatedDatabaseName, _f, self.DUT, self.Campaign)
            if 'nan' in _human_bID:
                _human_bID = _human_bID.remove('nan')
            if _human_bID is not None:
                match = list(set(_AI_bID).intersection(_human_bID))
                not_in_AI = list(set(_human_bID) - set(_AI_bID))
                not_in_HU = list(set(_AI_bID) - set(_human_bID))
            else:
                nbr_normal_human_clean = nbr_normal_human_clean + 1
                match = []
                not_in_AI = []
                not_in_HU = _AI_bID
            _matches += len(match)
            _fps += len(not_in_HU)
            _fns += len(not_in_AI)
        print()
        validated = _normal_validated + _anomalous_validated
        patches = validated*PATCHES
        print('A total of %s images have been validated for DUT %s in campaign %s.' % (validated, self.DUT, self.Campaign))
        print('Summary for the %s patches: ' % patches)
        tp = _matches
        fn = _fns
        fp = _fps
        tn = patches - _matches - _fps - _fns
        self.printCM(tp, tn, fp, fn)
        print()
        print('Summary for the %s whole images: ' % (validated))
        tp_full = _anomalous_validated - nbr_anomalous_human_clean
        fn_full = _normal_validated - nbr_normal_human_clean
        fp_full = nbr_anomalous_human_clean
        tn_full = nbr_normal_human_clean
        self.printCM(tp_full, tn_full, fp_full, fn_full)
        print()
        return _matches, _fns, _fps, tn, tp_full, fn_full, fp_full, tn_full

    def printValidationStatus_verbose(self):
        _annotated_files = self.get_fileNames(self.DBLocation, self.databaseName, self.DUT, self.Campaign, None)
        _validated_files = self.get_fileNames(self.DBLocation, self.validatedDatabaseName, self.DUT, self.Campaign, None)
        if _annotated_files is None:
            _annotated_files = []
        if _validated_files is None:
            _validated_files = []
        if len(_annotated_files) > 0:
            _not_annotated_files = list(set(_annotated_files) - set(_validated_files))
            print()
            print('Validation summary for directory: ', self.fpath)
            print('-------------------------------------------------------')
            print('Number of anomalous images:       ', len(_annotated_files))
            _val = len(_annotated_files) - len(_not_annotated_files)
            print('  of those validated:              %s (%s %%)' % (_val, _val*100/len(_annotated_files)))
            print('-------------------------------------------------------')
            return _not_annotated_files
        else:
            return []

    def printValidationStatus(self):
        _annotated_files = self.get_fileNames(self.DBLocation, self.databaseName, self.DUT, self.Campaign, None)
        _validated_files = self.get_fileNames(self.DBLocation, self.validatedDatabaseName, self.DUT, self.Campaign, None)
        if _annotated_files is None:
            _annotated_files = []
        if _validated_files is None:
            _validated_files = []
        if len(_annotated_files) > 0:
            _not_annotated_files = list(set(_annotated_files) - set(_validated_files))
            return _not_annotated_files
        else:
            return []

from tensorflow import keras
import tensorflow as tf
from utility.keras import setTrainable
class AIImageAnnotateEvent(ImageAnnotateEvent):
    def __init__(self, _wx=PATCHSIZE, _wy=PATCHSIZE):
        super().__init__(_wx, _wy)
        self.classifier = None
        self.background_remover = None
        self.threshold = 0.5
        self.anomalous_scan_steps = ''
        self.anomalous_scan_steps_npy = []
        self.count = 0
        self.saved_scan_steps = []
        self.background_threshold = 0.5
        self.encoder = None
        self.decoder = None
        self.image_decoded = None
        self.verbose = False

    def appendSavedStep(self,_scan_step):
        self.saved_scan_steps = np.append(self.saved_scan_steps, _scan_step)

    # function for loading classifier
    def loadClassifierModel(self, fpath, summary = False):
        self.classifier = keras.models.load_model(fpath)
        if self.verbose: print("  Loaded model from", fpath)
        if summary:
            print(self.classifier.summary())
        setTrainable(self.classifier, False)

    def loadBRModel(self, fpath, summary = False):
        self.background_remover = keras.models.load_model(fpath)
        if self.verbose: print("  Loaded model from", fpath)
        if summary:
            print(self.background_remover.summary())
        setTrainable(self.background_remover, False)

    def loadAE(self, fpath, summary = False):
        encoder_path = fpath + "_encoder"
        self.encoder = keras.models.load_model(encoder_path)
        if self.verbose: print("  Loaded encoder model from", encoder_path)
        if summary:
            print(self.encoder.summary())
        setTrainable(self.encoder, False)

        decoder_path = fpath + "_decoder"
        self.decoder = keras.models.load_model(decoder_path)
        if self.verbose: print("  Loaded decoder model from", decoder_path)
        if summary:
            print(self.decoder.summary())
        setTrainable(self.decoder, False)

    def setThreshold(self, _value=0.5):
        if self.verbose: print('Threshold set to %s.' % _value)
        self.threshold = _value

    def setVerbosity(self, verbose):
        self.verbose = verbose

    def setGrid(self, grid):
        self.grid = grid

    def set_img(self, array):
        super().set_img(array)

    def action(self,  step, pad):
        super().action()
        self.evaluateConvertedImage(step, pad)

    def divide_img_blocks(self, img, patchsize, patches):
        INPUT_DIM = tf.shape(img)[-1]
        split_img = tf.image.extract_patches(images=img, sizes=[1, patchsize, patchsize, 1], strides=[1, patchsize, patchsize, 1], rates=[1, 1, 1, 1], padding='VALID')
        batch = tf.reshape(split_img, [patches, patchsize, patchsize, INPUT_DIM])
        return batch.numpy().astype(np.int16)

    def divide_img_blocks_2(self, img, patchsize, patches):
        INPUT_DIM = tf.shape(img)[-1]
        patches = int((16)*(23))
        img = tf.image.crop_to_bounding_box(img, offset_height=80, offset_width=80, target_height=16*160, target_width=23*160)
        split_img = tf.image.extract_patches(images=img, sizes=[1, patchsize, patchsize, 1], strides=[1, patchsize, patchsize, 1], rates=[1, 1, 1, 1], padding='VALID')
        batch = tf.reshape(split_img, [patches, patchsize, patchsize, INPUT_DIM])
        return batch.numpy().astype(np.int16)

    def crop_img_bottom(self, img, pixels):
        return img[:, :-pixels, :]

    def evaluateConvertedImage(self, step, pad, _im=None):
        if _im == None:
            _image = np.expand_dims(self.img_array, axis=0)
        else:
            _image = np.expand_dims(_im, axis=0)

        _image = self.crop_img_bottom(_image, 16)

        _orig_image = np.expand_dims(_image, axis=3).astype(np.int16)
        _cropped_orig_images = self.divide_img_blocks(_orig_image, PATCHSIZE, PATCHES)

        _background = np.squeeze(self.background_remover(_cropped_orig_images))
        _background_patches = np.array((_background > self.background_threshold))
        _background_remover = 1-_background_patches

        z = self.encoder(_image)
        self.image_decoded = np.array(self.decoder(z)).astype(np.uint8)

        # compute difference
        _image = np.expand_dims(_image, axis=3).astype(np.int16)
        _image = np.abs(_image - self.image_decoded.astype(np.int16))

        # crop images and evaluate all individually
        _cropped_images = self.divide_img_blocks(_image, PATCHSIZE, PATCHES)

        _classifier = np.squeeze(self.classifier(_cropped_images))
        annotations = (_classifier > self.threshold)
        annotations = np.multiply(annotations, _background_remover)
        annotations = np.transpose(annotations.nonzero())

        annotations_2 = []
        if len(annotations)>0:
            self.anomalous_scan_steps = self.anomalous_scan_steps+str(step)+' '
            self.anomalous_scan_steps_npy = np.append(self.anomalous_scan_steps_npy, step)

            self.setFilePath(os.path.join(self.fpath, "annotated_image_pad%i_step%i" % (pad, step)))
            if self.verbose: print('Anomaly detected in image: annotated image will be saved to %s, please check.\n' % self.fpath)
            if self.verbose: print('Annotation index: \n %s' % annotations.flatten())
            self.count += 1
            update_canvas = False
            self.save_to_database = True
            for _annotation in annotations:
                p = _annotation[0]
                update_canvas = self.addAnnotation_AI(p, 1) or update_canvas

            if update_canvas:
                self.updateFigure()
            self.filename = self.filename.replace("initial_scan", "annotated_image")
            self.processAnnotations()
            self.removeAllAnnotations()

        elif len(annotations) == 0 and self.grid == 2:
            if self.verbose: print("Evaluation with grid 2.")
            ## if image deemed clean by grid 1, check with grid 2
            _cropped_images_2 = self.divide_img_blocks_2(_image, PATCHSIZE, PATCHES)
            _classifier_2 = np.squeeze(self.classifier(_cropped_images_2))
            annotations_2 = (_classifier_2 > self.threshold)
            #annotations_2 = np.multiply(annotations_2, _background_remover)
            annotations_2 = np.transpose(annotations_2.nonzero())

            if len(annotations_2) > 0:
                self.count += 1
                self.anomalous_scan_steps = self.anomalous_scan_steps + str(step) + ' '
                self.anomalous_scan_steps_npy = np.append(self.anomalous_scan_steps_npy, step)
                self.setFilePath(os.path.join(self.fpath, "annotated_image_pad%i_step%i" % (pad, step)))
                if self.verbose: print('GRID 2: Anomaly detected in image: annotated image will be saved to %s, please check.\n' % self.fpath)
                if self.verbose: print('Annotation index: \n %s' % annotations_2.flatten())
                update_canvas = False
                self.save_to_database = True
                for _annotation in annotations_2:
                    p = _annotation[0]
                    update_canvas = self.addAnnotation_AI(p, 2) or update_canvas

                if update_canvas:
                    self.updateFigure()
                self.filename = self.filename.replace("initial_scan", "annotated_image")
                self.processAnnotations()
                self.removeAllAnnotations()

        if len(annotations) == 0 and len(annotations_2) == 0 and self.verbose:
            print('Clean image!\n')
