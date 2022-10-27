# Created by Sonja Grönroos in July 2022
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
import pathlib
import argparse

from glob import glob1
from beautifultable import BeautifulTable
from classes.ImageHandler_v2 import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, help="Select mode: VALIDATE to validate new AI scanned sensors/SHOW to show summary of model performance based on previous validations.",
                    required=True)
parser.add_argument("--CampaignName", type = str,
                    help="Name of campaign.", required = True)
parser.add_argument("--DUTName", type=str,
                    help="(option) Name of DUT.", required = False, default = None)
args = parser.parse_args()

mode = args.mode
campaign_to_analyse = args.CampaignName
dut = args.DUTName

## checking that campaign and dut exist
if campaign_to_analyse in os.listdir('outputs'):
    output_dir = os.path.abspath(os.path.join("outputs", campaign_to_analyse))
    print('Analysing campaign %s' % campaign_to_analyse)
else:
    print('INPUT ERROR: campaign does not exist. Please check input.')
    exit()

all_duts_in_dir = glob.glob(os.path.join(output_dir, "*"))
if all_duts_in_dir is None:
    print("No DUTs found. Exiting.")
    exit()
if dut is not None:
    if os.path.join(output_dir, dut) in all_duts_in_dir:
        print('Selected DUT %s' % dut)
    else:
        print('INPUT ERROR: dut does not exist. Please check input.')
        exit()

overwrite = False
while True:
    if mode.casefold() not in ['validate'.casefold(), 'show'.casefold()]:
        print('INPUT ERROR: Mode is not valid. Please check arguments.')
        break
    if mode.casefold() == 'validate'.casefold():
        print("Initializing validation program. \n")
        print('Reading directory %s.' % output_dir)
        print()
        not_val_duts = []
        validation_check = ImageValidateEvent()
        for _dut in all_duts_in_dir:
            validation_check.set_DUT(os.path.basename(os.path.normpath(_dut)))
            validation_check.set_Campaign(campaign_to_analyse)
            not_ann = validation_check.printValidationStatus()
            if len(not_ann) > 0:
                not_val_duts = np.append(not_val_duts, pathlib.PurePath(_dut).name)
        if dut == None:
            print("You have not specified a DUT in the input.")
            print('There are %s DUTs, %s of which contain annotated images that have not been validated:' % (len(all_duts_in_dir), len(not_val_duts)))
            _dl, _sl = [], []
            for i in range(len(all_duts_in_dir)):
                dut = pathlib.PurePath(all_duts_in_dir[i]).name
                ann_npys = len(glob1(os.path.join(output_dir, dut), "ann*.npy"))
                if (dut in not_val_duts) and (ann_npys > 0):
                    _dl = np.append(_dl, dut)
                    _sl = np.append(_sl, ann_npys)
                elif (dut not in not_val_duts) and (ann_npys > 0):
                    _dl = np.append(_dl, dut)
                    _sl = np.append(_sl, 0)
                elif ann_npys == 0:
                    _dl = np.append(_dl, dut)
                    _sl = np.append(_sl, "False")
                col_headers = ["x", "y"]

            table = BeautifulTable()
            table.columns.append(_dl, header="DUT Name")
            table.columns.append(_sl, header="To validate")
            print(table)

            if len(not_val_duts) == 0:
                print('Every DUT has been validated already.')
                not_val_duts_overwrite = []
                while True:
                    answer = input('Will you overwrite previous validations (yes/no)? \n')
                    if answer == 'yes':
                        dut_to_overwrite = input('Give name of DUT to overwrite or type "all". Write "END" to exit. \n')
                        overwrite = True
                        if dut_to_overwrite.casefold() == "all".casefold():
                            not_val_duts_overwrite = all_duts_in_dir
                            break
                        elif dut_to_overwrite == "END":
                            exit()
                        elif os.path.join(output_dir, dut_to_overwrite) in all_duts_in_dir:
                            print("DUT exists")
                            not_val_duts_overwrite = np.append(not_val_duts_overwrite, dut_to_overwrite)
                            break
                        else:
                            print("Try again!")
                    elif answer == 'no':
                        print('Nothing to do, exiting.')
                        exit()
                    print('Try again!')
                to_val = not_val_duts_overwrite
                to_val_filtered = []
            print()
            if len(not_val_duts) > 0:
                while True:
                    to_val = input('Type "all" to validate all non-validated DUTs OR type the name of the DUT name you wish to validate. Write "END" to exit. \n')
                    if to_val.casefold() == 'All'.casefold():
                        to_val = not_val_duts
                        break
                    elif os.path.join(output_dir,  to_val) in all_duts_in_dir:
                        if to_val not in not_val_duts:
                            print("You will overwrite old validations.")
                            overwrite = True
                        to_val = [to_val]
                        break
                    elif to_val == "END":
                        exit()
                    else:
                        print('Error: chosen DUT does not exist, try again.')
                to_val_filtered = []
        elif dut is not None:
            overwrite = True
            to_val = [dut]
            to_val_filtered = []
        print()
        for _dut in to_val:
            ann_npys = len(glob1(os.path.join(output_dir, _dut), "ann*.npy"))
            if ann_npys > 0:
                to_val_filtered = np.append(to_val_filtered, _dut)
        print()

        for dut in to_val_filtered:
            print('Analysing DUT ', dut)
            print("  Overwriting: %s." % overwrite)
            ann_npys = len(glob1(os.path.join(output_dir, _dut), "ann*.npy"))
            init_npys = len(glob1(os.path.join(output_dir, _dut), "init*.npy"))
            print('  Contains %s annotated images to validate. Additionally, you will validate %s normal images.' % (ann_npys, init_npys))
            validation = ImageValidateEvent()
            plt.close(1)
            validation.set_DUT(os.path.basename(os.path.normpath(dut)))
            validation.set_Campaign(campaign_to_analyse)
            validation.setFilePath(os.path.join(output_dir, dut))
            validation.set_frac(1)
            validation.setAI(False)
            validation.validation_action('all', overwrite)
            print()
            validation.createValidationSummary()
        break

    if mode.casefold() == 'show'.casefold():
        patch_tps, patch_fns, patch_fps, patch_tns = [], [], [], []
        full_tps, full_fns, full_fps, full_tns = [], [], [], []
        show = ImageValidateEvent()
        plt.close(1)
        validation_database = pd.read_pickle(os.path.abspath(os.path.join(show.DBLocation, show.validatedDatabaseName))).reset_index()
        campaigns = pd.unique(validation_database.Campaign)
        val_db = validation_database[validation_database.Campaign == campaign_to_analyse]
        annotation_db = validation_database[validation_database.Campaign == campaign_to_analyse]
        if dut is None:
            validated_DUTs = pd.unique(val_db.DUT)
        elif dut is not None and os.path.join(output_dir,dut) in all_duts_in_dir:
            print("Analysing DUT %s" % dut)
            validated_DUTs = [dut]
        elif dut is not None and os.path.join(output_dir,dut) not in all_duts_in_dir:
            print("DUT %s not in directory!" % dut)
            exit()
        for dut in validated_DUTs:
            print('DUT in validation DB: ', dut)
            if os.path.join(output_dir,dut) not in all_duts_in_dir:
                print("DUT not in directory!")
                print("-----------------------------------------------------------------------------------------")
                print()
                continue
            show.set_DUT(os.path.basename(os.path.normpath(dut)))
            show.set_Campaign(campaign_to_analyse)
            not_ann = show.printValidationStatus()
            if len(not_ann) == 0:
                print("All annotated files have been validated.")
            elif len(not_ann) != 0:
                print("%s annotated files have not been validated." % len(not_ann))
            show.set_DUT(os.path.basename(os.path.normpath(dut)))
            show.set_Campaign(campaign_to_analyse)
            _patch_tps, _patch_fns, _patch_fps, _patch_tns, _full_tps, _full_fns, _full_fps, _full_tns  = show.createValidationSummary()

            patch_tps = np.append(patch_tps, _patch_tps)
            patch_fns = np.append(patch_fns, _patch_fns)
            patch_fps = np.append(patch_fps, _patch_fps)
            patch_tns = np.append(patch_tns, _patch_tns)

            full_tps = np.append(full_tps, _full_tps)
            full_fns = np.append(full_fns, _full_fns)
            full_fps = np.append(full_fps, _full_fps)
            full_tns = np.append(full_tns, _full_tns)
            print("-----------------------------------------------------------------------------------------")
            print()
        print('Sum for whole campaign: patches')
        show.printCM(int(np.sum(patch_tps)), int(np.sum(patch_tns)), int(np.sum(patch_fps)), int(np.sum(patch_fns)))
        print()
        print('Sum for whole campaign: whole images')
        show.printCM(int(np.sum(full_tps)), int(np.sum(full_tns)), int(np.sum(full_fps)), int(np.sum(full_fns)))
        break