# Created by Thorben Quast, August 2021
# Modified by Sonja Grönroos, August 2022
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
import math
import datetime
import pandas as pd
import numpy as np

from utility.dataset_helpers import open_db
from config import *

def read_AI_indices_from_prints(output_dir, normal):
    f = open(os.path.join(output_dir, "run_PS_prints.txt"), 'r')
    if not normal:
        search = 'Anomalous scan indices'
    elif normal:
        search = 'Normal scan indices'
    read_next = False
    for line in f:
        if read_next:
            search_result = line
        if search in line:
            read_next = True
    f.close()
    try:
        AI_scan_steps = np.array(list(map(int, search_result.split(' ')[:-1])))
        print(AI_scan_steps)
        return np.unique(AI_scan_steps)
    except:
        print("Scan indexes could not be read. \n")
        return []

def read_AI_indices(output_dir):
    f = open(os.path.join(output_dir, "annotation_indices.txt"), 'r')
    last_line = f.readline()
    f.close()
    try:
        AI_scan_steps = np.array(list(map(int, last_line.split(' ')[:-1])))
        return np.unique(AI_scan_steps)
    except:
        print("AI annotation scan indexes could not be read. \n")
        return []

def create_dir(_dir):
	if not os.path.exists(_dir):
		mother_dir = os.path.dirname(_dir)
		if not os.path.exists(mother_dir):
			create_dir(mother_dir)
		os.mkdir(_dir)
	return _dir

## for grid 1, original
def box_index_to_coords(box_index):
    row = math.floor(box_index / 24)
    col = box_index % 24
    return col * 160, row * 160

## for grid 2, secondary
def box_index_to_coords_2(box_index):
    row = math.floor(box_index / (24-1))
    col = box_index % (24-1)
    return (col * 160)+80, (row * 160)+80

def str_to_dt(str):
    return datetime.datetime.strptime(str, '%Y-%m-%d').date()

def filter_db(db, date_filter):
    return db[db.Date <= date_filter]

def open_filter_db(name, date_filter):
    f = os.path.join(TRAIN_DIR_LOC, name)  # newest database
    with pd.HDFStore(f, mode='r') as store:
        db = store.select('db')
        print(f'Reading {f}')
    return filter_db(db, date_filter)

def check_new_train_data(newest_val_date):
    train_db = open_db(os.path.join(TRAIN_DIR_LOC, 'TRAIN_DATABASE'))
    newest_train_date = train_db.Date.unique().max()
    if newest_val_date > newest_train_date:
        return True,  newest_train_date
    else:
        return False, newest_train_date

def check_new_test_data(model_date):
    test_db = open_db(os.path.join(TRAIN_DIR_LOC, 'TEST_DATABASE'))
    newest_test_date = test_db.Date.unique().max()
    if model_date < newest_test_date:
        new_test_dates = test_db[test_db.Date > model_date].Date.unique()
        return True, new_test_dates
    else:
        return False, newest_test_date

