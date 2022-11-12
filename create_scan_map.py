# Created by Sonja Grönroos, August 2022
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

import math
import numpy as np
import pandas as pd
import os
import argparse

from matplotlib import pyplot as plt
from classes.ScanMapping import ScanMapping

parser = argparse.ArgumentParser()
parser.add_argument("--SensorType", type=str, help="LD or HD", required=True)
parser.add_argument("--Partial", type=str, help="Name of partial cut or NONE", required=True)
args = parser.parse_args()

## function to set the slope of left side
def left_fit(y, partial, sensortype):
    if sensortype == 'LD' and partial.casefold() == 'Right'.casefold():
        return LD_right_cut
    if sensortype == 'HD' and partial.casefold() == 'Right'.casefold():
        return HD_right_cut
    if sensortype == 'LD' and partial.casefold() == 'Three'.casefold():
        return LD_three_cut
    else:
        return (y-175.8)/1.776

## funcion to set the slope of the right side
def right_fit(y, partial, sensortype):
    if sensortype == 'LD' and partial.casefold() == 'Left'.casefold():
        return LD_left_cut
    if sensortype == 'HD' and partial.casefold() == 'Left'.casefold():
        return HD_left_cut
    if sensortype == 'LD' and partial.casefold() == 'Five'.casefold():
        return LD_five_cut
    if sensortype == 'HD' and partial.casefold() == 'Five'.casefold():
        return HD_right_cut
    else:
        return ((y-175.8)/1.776)*(-1)

## empirically determined cuts for the partial mappings
LD_left_cut = 5
LD_right_cut = -5
HD_left_cut = -30
HD_right_cut = 30
LD_three_cut = 47
LD_five_cut = 55
LD_top_cut = -5
LD_bottom_cut = 5
HD_bottom_cut = 25
HD_top_cut = 34

## these are the extremeties of the produced Hexagonal scan pattern
top_left = [-50, 87]
top_right = [50, 87]
center_left = [-99, 0.0]
center_right = [99, 0.0]

## scan step lengths, are empirically determined to get optimal overlap of images
dy = 7.6
dx = 10

## empirically determined shifts for the partial mappings (for centering)
maps_to_shift = ['LD_Top', 'LD_Bottom', 'LD_Left', 'LD_Right', 'LD_Three', 'HD_Top', 'HD_Left', 'HD_Right', 'HD_Five']
x_shifts = [0, 0, center_right[0]/2 , -center_right[0]/2, -(2*center_right[0])/3, 0, (2*center_right[0])/3, -(2*center_right[0])/3, (center_right[0])/3]
y_shifts = [-top_left[1]/2, top_left[1]/2, 0,  0, 0,  -2*(top_left[1]/3), 0, 0, 0]
shifts = pd.DataFrame(list(zip(maps_to_shift, x_shifts, y_shifts)), columns =['map_name', 'x_shift', 'y_shift']).set_index('map_name')

sensortype = args.SensorType.upper()
partial = args.Partial.capitalize()

if sensortype not in ['LD', 'HD']:
    print('Incorrect SensorType. Try again.')
    exit()
if partial not in ['Left', 'Right', 'Full', 'Three', 'Five', 'Bottom', 'Top']:
    print('Incorrect partial name. Try again.')
    exit()
if sensortype == 'HD' and partial == 'Three':
    print('Incorrect arguments. Try again.')
    exit()

name = sensortype+'_'+partial

## define parameters for different partial cuts
if sensortype == 'LD':
    if partial.casefold() == "Left".casefold():
        top_right = [LD_left_cut, top_left[1]]
        center_right = [LD_left_cut, center_left[1]]
    if partial.casefold() == 'Right'.casefold():
        top_left = [LD_right_cut, top_left[1]]
        center_left = [LD_right_cut, center_left[1]]
    if partial.casefold() == 'Three'.casefold():
        top_left = [LD_three_cut, top_left[1]]
        three_images_per_row = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6]
    if partial.casefold() == 'five'.casefold():
        top_right = [LD_five_cut, top_left[1]]
if sensortype == 'HD':
    if partial.casefold() == "Left".casefold():
        top_right = [HD_left_cut, top_left[1]]
        center_right = [HD_left_cut, center_left[1]]
    if partial.casefold() == 'Right'.casefold():
        top_left = [HD_right_cut, top_left[1]]
        center_left = [HD_right_cut, center_left[1]]
        three_images_per_row = [1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7]
    if partial.casefold() == 'five'.casefold():
        top_right = [HD_right_cut, top_left[1]]

## get sensortype pad locations
sm = ScanMapping()
if sensortype == 'HD':
    sm.geofile = os.path.abspath("maps/hex_positions_HPK_432ch_8inch_edge_ring_testcap.txt")
    sm.NPads = 444
elif sensortype == 'LD':
    sm.geofile = os.path.abspath("maps/hex_positions_HPK_198ch_8inch_edge_ring_testcap.txt")
    sm.NPads = 198

sm.loadGeoFile()
pad_locations = sm.map
pad_coords = pad_locations[['x_mm','y_mm']].to_numpy()
left = top_left
right = top_right
y = top_right[1]

xs, ys = [], []
pad, guard_ring = [], []
order, index = [], []

i, j = 0, 0
while y > 0:
    if partial.casefold() != "Three".casefold():
        images_per_row = math.ceil(np.sqrt((left[0] - right[0])**2) / dx)
        dx = (np.sqrt((left[0] - right[0])**2))/images_per_row
        k = 0
        ## not optimal optimizing
        while k < 10:
            if dx < 9:
                images_per_row = images_per_row - 1
                dx = (np.sqrt((left[0] - right[0]) ** 2)) / (images_per_row)
            elif dx > 10:
                images_per_row = images_per_row + 1
                dx = (np.sqrt((left[0] - right[0]) ** 2)) / (images_per_row)
            k = k + 1
    if sensortype == 'LD' and partial.casefold() == "Three".casefold():
        images_per_row = three_images_per_row[j]
        dx = abs((abs(right[0]) - abs(left[0]))/images_per_row)
    j = j + 1
    x = left[0]
    for image in range(images_per_row+1):
        coordinate = (x, y)
        distances = np.linalg.norm(pad_coords - coordinate, axis=1)
        min_index = np.argmin(distances)
        pad = np.append(pad, pad_locations.iloc[min_index].padnr)
        xs = np.append(xs, x)
        ys = np.append(ys, y)
        i = i + 1
        if y >= top_left[1]-dy:
            guard_ring = np.append(guard_ring, 1)
            gr = 1
        elif image == 0 or image == 1:
            guard_ring = np.append(guard_ring, 1)
            gr = 1
        elif image == images_per_row or image == images_per_row - 1:
            guard_ring = np.append(guard_ring, 1)
            gr = 1
        else:
            guard_ring = np.append(guard_ring, 0)
            gr = 0
        ## create bottom part
        if y > 0 and y <= top_left[1]:
            coordinate2 = (-x, -y)
            distances = np.linalg.norm(pad_coords - coordinate2, axis=1)
            min_index = np.argmin(distances)
            pad = np.append(pad, pad_locations.iloc[min_index].padnr)
            xs = np.append(xs, x)
            ys = np.append(ys, -y)
            i = i+ 1
            if -y <= -top_left[1]+dy:
                gr = 1
            guard_ring = np.append(guard_ring, gr)
        x = x + dx
    y = y - dy
    left = [left_fit(y, partial, sensortype), y]
    right = [right_fit(y, partial, sensortype), y]

## define dataframe, remove potential duplicates
print("Created a scan pattern with %s scan points." % len(xs))
full_map_df = pd.DataFrame({'padnumber': pad, 'xposition': xs, 'yposition': ys, 'guard_ring': guard_ring})
full_map_df = full_map_df.astype({"padnumber": int, "xposition": float, "yposition": float, "guard_ring": int})
full_map_df = full_map_df.drop_duplicates(subset = ['xposition', 'yposition'], keep='first').sort_values(['yposition', 'xposition'], ascending=[False, True])

## cutting partial scans
if partial is not None and sensortype == 'LD':
    if partial.casefold() == "Top".casefold():
        full_map_df = full_map_df.loc[full_map_df['yposition'] > LD_top_cut]
        full_map_df.loc[full_map_df['yposition'] < 4, 'guard_ring'] = 1
    if partial.casefold() == "Bottom".casefold():
        full_map_df = full_map_df.loc[full_map_df['yposition'] < LD_bottom_cut]
        full_map_df.loc[full_map_df['yposition'] > -4, 'guard_ring'] = 1
elif partial is not None and sensortype == 'HD':
    if partial.casefold() == "Top".casefold():
        full_map_df = full_map_df.loc[full_map_df['yposition'] > HD_bottom_cut]
        full_map_df.loc[full_map_df['yposition'] < 34, 'guard_ring'] = 1
    if partial.casefold() == "Bottom".casefold():
        full_map_df = full_map_df.loc[full_map_df['yposition'] < HD_top_cut]
        full_map_df.loc[full_map_df['yposition'] > 26, 'guard_ring'] = 1

## apply shifts
if name in maps_to_shift:
    x_shift = shifts.loc[name].x_shift
    y_shift = shifts.loc[name].y_shift
    full_map_df['xposition'] = full_map_df['xposition'] + x_shift
    full_map_df['yposition'] = full_map_df['yposition'] + y_shift

## add extra guard rings
if name == 'LD_Left' :
    full_map_df.loc[np.abs(full_map_df['yposition']) < 4, "guard_ring"] = 1
if name == 'LD_Right':
    full_map_df.loc[np.abs(full_map_df['yposition']) < 4, "guard_ring"] = 1
    full_map_df.loc[full_map_df['xposition'].between(-5,5), "guard_ring"] = 1
if name == 'LD_Bottom' or name == 'LD_Top':
    full_map_df.loc[np.abs(full_map_df['xposition']).between(-5,5), "guard_ring"] = 1
    full_map_df.loc[full_map_df['xposition'].between(40, 50), "guard_ring"] = 1
if name == 'LD_Five':
    full_map_df.loc[np.abs(full_map_df['yposition']).between(-5,5), "guard_ring"] = 1
    full_map_df.loc[np.abs(full_map_df['xposition']).between(-5,5), "guard_ring"] = 1
if name == 'LD_Three':
    full_map_df.loc[np.abs(full_map_df['yposition']).between(-5,5), "guard_ring"] = 1
    full_map_df.loc[np.abs(full_map_df['xposition']).between(-5,5), "guard_ring"] = 1
if name == 'LD_Five':
    full_map_df.loc[np.abs(full_map_df['yposition']).between(-3,3), "guard_ring"] = 1
if name == 'HD_Top':
    full_map_df.loc[full_map_df['xposition'].between(-40,-20), "guard_ring"] = 1
    full_map_df.loc[full_map_df['xposition'].between(20, 40), "guard_ring"] = 1
if name == 'HD_Left' or name == 'HD_Right':
    full_map_df.loc[full_map_df['yposition'].between(10,30), "guard_ring"] = 1
if name == 'HD_Five':
    full_map_df.loc[full_map_df['yposition'].between(10, 30), "guard_ring"] = 1
    full_map_df.loc[full_map_df['xposition'].between(-6, 6), "guard_ring"] = 1



## define scans photo-taking order
order = []
ylist = np.unique(full_map_df.yposition.tolist())[::-1]
last, row = 0, 2
for y in ylist:
    xsl = full_map_df[full_map_df.yposition == y].xposition.tolist()
    lxs = len(xsl)
    row_indices = np.arange(last+1, last+1+lxs)
    last = row_indices[-1]
    if row % 2 == 0:
        row_indices = row_indices
    else:
        row_indices = np.flip(row_indices)
    order = np.append(order, row_indices)
    row = row + 1

## save scan map
full_map_df["scan_order"] = order
full_map_df = full_map_df.astype({"scan_order": int})
full_map_df.to_csv(r'maps/custom_map_%s.txt' % name, header=None, index=None, sep='\t', mode='w')

guard_ring_df = full_map_df[full_map_df.guard_ring == 1]

## visualize created scan map
plt.scatter(full_map_df.xposition.tolist(), full_map_df.yposition.tolist(), zorder = 2, label='Scan point')
plt.scatter(guard_ring_df.xposition.tolist(), guard_ring_df.yposition.tolist(), zorder = 3, color = 'red', label = 'Guard ring scan point')
plt.grid(zorder = -3)
plt.ylim(-100, 110)
plt.xlim(-120, 120)
plt.legend()
plt.title("Created scan pattern %s" % name)
plt.savefig('maps/images/%s.png' % name, dpi=400)
plt.show()