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

from matplotlib import pyplot as plt
from scipy import spatial
from classes.ScanMapping import ScanMapping

def left_fit(y):
    return (y-175.8)/1.776

def right_fit(y):
    return ((y-175.8)/1.776)*(-1)

## these are the "extremeties" of the produced Hexagonal scan pattern
top_left = [-50, 87]
top_right = [50, 87]
center_left = [-99, 0.0]
center_right = [99, 0.0]

### USER NEEDS TO SET THESE ###
## scan step lengths, need to be empirically determined to get optimal overlap of images
#dx = 9.5
dy = 7.6
dx = 10

sensortype = 'LD'
name = sensortype+'_2patch_overlap_385'
### USER OPTIONS END ###

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
print(pad_locations)
pad_coords = pad_locations[['x_mm','y_mm']].to_numpy()
print(pad_coords)
left = top_left
right = top_right
y = top_right[1]

xs, ys = [], []
pad = []
guard_ring = []
order, index = [], []

i = 0
while y > 0:
    images_per_row = math.ceil((abs(left[0]) + abs(right[0])) / dx)
    dx = (abs(left[0]) + abs(right[0]))/images_per_row
    x = left[0]
    for image in range(images_per_row+1):
        coordinate = (x, y)
        distances = np.linalg.norm(pad_coords - coordinate, axis=1)
        min_index = np.argmin(distances)
        pad = np.append(pad, pad_locations.iloc[min_index].padnr)
        xs = np.append(xs, x)
        ys = np.append(ys, y)
        i = i + 1
        if y == top_left[1]:
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
        if y > 0 and y < top_left[1]:
            coordinate2 = (-x, -y)
            distances = np.linalg.norm(pad_coords - coordinate2, axis=1)
            min_index = np.argmin(distances)
            pad = np.append(pad, pad_locations.iloc[min_index].padnr)
            xs = np.append(xs, -x)
            ys = np.append(ys, -y)
            i = i+ 1
            if -y == -top_left[1]+dy:
                gr = 1
            guard_ring = np.append(guard_ring, gr)
        x = x + dx
    y = y - dy
    left = [left_fit(y), y]
    right = [right_fit(y), y]

print("Created a scan pattern with %s scan points." % len(xs))
map_df = pd.DataFrame({ 'padnumber': pad, 'xposition': xs, 'yposition': ys, 'guard_ring': guard_ring})
map_df = map_df.astype({ "padnumber": int, "xposition": float, "yposition": float, "guard_ring": int})
map_df = map_df.drop_duplicates(subset = ['xposition','yposition'], keep='first').sort_values(['yposition', 'xposition'], ascending=[False, True])

order = []
ylist = np.unique(map_df.yposition.tolist())[::-1]
last = 0
row = 2
for y in ylist:
    xsl = map_df[map_df.yposition == y].xposition.tolist()
    lxs = len(xsl)

    row_indices = np.arange(last+1 , last+1+lxs)
    last = row_indices[-1]
    if row % 2 == 0:
        row_indices= row_indices
    else:
        row_indices = np.flip(row_indices)
    order = np.append(order, row_indices)
    row = row + 1

## save scan map
map_df["scan_order"] = order
map_df = map_df.astype( {"scan_order": int})
print(map_df)
map_df.to_csv(r'maps/custom_map_%s.txt' % name, header=None, index=None, sep='\t', mode='w')

## visualize created scan map
plt.scatter(xs,ys, zorder = 2)
plt.grid(zorder = -3)
plt.title("Created scan pattern %s" % name)
plt.show()