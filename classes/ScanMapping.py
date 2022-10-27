# Created by Thorben Quast, August 2021
# Modified by Sonja Grönroos in August 2022
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

import os, time
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

GEOFILES = {
    "HPK_198ch_8inch": os.path.abspath("maps/hex_positions_HPK_198ch_8inch_edge_ring_testcap.txt"),
    "HPK_432ch_8inch": os.path.abspath("maps/hex_positions_HPK_432ch_8inch_edge_ring_testcap.txt"),
    "custom_map_LD": os.path.abspath("maps/custom_map_LD_2patch_overlap.txt"),
    "custom_map_HD": os.path.abspath("maps/custom_map_HD_2patch_overlap.txt"),
    "custom_map_LD_396": os.path.abspath("maps/custom_map_LD_2patch_overlap_396.txt"),
    "custom_map_HD_396": os.path.abspath("maps/custom_map_HD_2patch_overlap_396.txt"),
    "custom_map_LD_385": os.path.abspath("maps/custom_map_LD_2patch_overlap_385.txt"),
    "custom_map_HD_385": os.path.abspath("maps/custom_map_HD_2patch_overlap_385.txt")
}

NPADS = {
    "HPK_198ch_8inch": 198,
    "custom_map_HD": 444,
    "custom_map_LD": 198,
    "custom_map_LD_396": 198,
    "custom_map_HD_396": 444,
    "custom_map_LD_385": 198,
    "custom_map_HD_385": 444,
    "HPK_432ch_8inch": 444
}

# not needed for custom
OFFSETS_CM = {
    "HPK_198ch_8inch": (0.01, -0.43),
    "HPK_432ch_8inch": (0.0, 0.0)
}

# not needed for custom
DXDY_CM = {
    "HPK_198ch_8inch": (0.79, 0.69),
    "HPK_432ch_8inch": (0.54, 0.47)
}

SCALE = 10 * 8. / 5.  # scaling from hexplot mapping to real mm, empirically determined

class ScanMapping:
    def __init__(self, geo="HPK_432ch_8inch"):
        self.geometry = geo
        self.geofile = GEOFILES[geo]
        self.NPads = NPADS[geo]
        self.map = None
        self.guard_ring_scan_steps = None
        self.scan_order = None
        self.dx = None
        self.dy = None
        self.title = "Scan pattern"

        self.pattern = None
        self.scan_sequence = []
        self.scan_step = -1

        self.matplot_fig = None
        self.matplot_ax = None
        self.matplot_image = None
        self.matplot_fig, self.matplot_ax = None, None

        self.legend_elements = [Line2D([0], [0], marker='o', color='w', label='Unprocessed',markerfacecolor='b', markersize=10),
                                Line2D([0], [0], marker='o', color='w', label='Being processed', markerfacecolor='magenta', markersize=10),
                                Line2D([0], [0], marker='o', color='w', label='Processed', markerfacecolor='g', markersize=10),
                                Line2D([0], [0], marker='o', color='w', label='Pre-selected anomaly', markerfacecolor='r', markersize=10)]

    def loadGeoFile(self, fpath=None):
        if fpath != None:
            self.geofile = fpath
        # apply offsets
        # scaled to real mm
        if "HPK" in self.geofile:
            self.map = pd.DataFrame(
                np.genfromtxt(self.geofile, usecols=(0, 1, 2),
                              dtype=[("padnr", int), ("x_mm", float), ("y_mm", float)]))
            self.map = self.map[self.map.padnr <= self.NPads]
            self.map.x_mm = SCALE * (self.map.x_mm - OFFSETS_CM[self.geometry][0])
            self.map.y_mm = SCALE * (self.map.y_mm - OFFSETS_CM[self.geometry][1])

            self.dx = DXDY_CM[self.geometry][0] * SCALE
            self.dy = DXDY_CM[self.geometry][1] * SCALE
        if "custom" in self.geofile:
            self.map = pd.DataFrame(
                np.genfromtxt(self.geofile, usecols=(0, 1, 2),
                              dtype=[("padnr", int), ("x_mm", float), ("y_mm", float)]))
            self.scan_order = pd.DataFrame(np.genfromtxt(self.geofile, usecols=(4), dtype=[("scan_order", int)]))
            self.guard_ring, _ = np.where(pd.DataFrame(
            np.genfromtxt(self.geofile, usecols=(3), dtype=[("guard_ring", int)])).to_numpy() == 1)
            self.guard_ring_scan_steps = self.guard_ring + 1

    def openFigure(self):
        if self.matplot_fig == None:
            print("opening new figure")
            self.matplot_fig, self.matplot_ax = plt.subplots(figsize=(10, 7))
            self.matplot_fig.canvas.mpl_connect('close_event', lambda x: self.setNone())
            self.matplot_fig.show()
            plt.pause(0.01)

    def update_title(self, title):
        self.title = title

    def update_legend(self):
        self.legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Unprocessed', markerfacecolor='b', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Being processed', markerfacecolor='magenta', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Processed', markerfacecolor='g', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Pre-selected anomaly', markerfacecolor='r', markersize=10),
            Line2D([0], [0], marker='o', color='w', label='Not pre-selected anomaly', markerfacecolor="orange", markersize=10)]

    def closeFigure(self):
        if self.matplot_fig == None:
            self.matplot_fig.close()

    def setNone(self):
        self.matplot_fig, self.matplot_ax = None, None
        self.matplot_image = None

    def createPattern(self, detail_level=2):
        self.pattern = None
        self.pattern = self.map
        self.pattern = self.pattern.assign(detail_level=0)

        # force dataillevel=0 for custom maps
        if "custom" in self.geofile:
            detail_level = 0
            #self.pattern = self.pattern.assign(order=np.arange(1, self.map.shape[0]+1))
            self.pattern = self.pattern.assign(order=self.scan_order)
        else:
            self.pattern = self.pattern.assign(order=10 * self.pattern.padnr)

        if detail_level >= 1:
            add_patt = self.pattern[self.pattern.y_mm != 0]
            add_patt.order = add_patt.order + 1
            add_patt.detail_level = 1

            add_patt_upper_half = add_patt[add_patt.y_mm > 0]
            add_patt_upper_half.y_mm = add_patt_upper_half.y_mm - 2. * self.dy / 3.
            add_patt_lower_half = add_patt[add_patt.y_mm < 0]
            add_patt_lower_half.y_mm = add_patt_lower_half.y_mm + 2 * self.dy / 3.
            self.pattern = pd.concat([self.pattern, add_patt_upper_half, add_patt_lower_half])
        if detail_level >= 2:
            add_patt = self.pattern[self.pattern.order % 10 == 0]
            add_patt.detail_level = 2

            add_patt_q1 = add_patt[(add_patt.x_mm < 0) & (add_patt.y_mm > 0)]
            add_patt_q2 = add_patt[(add_patt.x_mm > 0) & (add_patt.y_mm > 0)]
            add_patt_q3 = add_patt[(add_patt.x_mm < 0) & (add_patt.y_mm < 0)]
            add_patt_q4 = add_patt[(add_patt.x_mm > 0) & (add_patt.y_mm < 0)]

            add_patt_q1.x_mm = add_patt_q1.x_mm + self.dx / 2
            add_patt_q1.y_mm = add_patt_q1.y_mm - 1. * self.dy / 3
            add_patt_q1.order = add_patt_q1.order + 2

            add_patt_q2.x_mm = add_patt_q2.x_mm - self.dx / 2
            add_patt_q2.y_mm = add_patt_q2.y_mm - 1. * self.dy / 3
            add_patt_q2.order = add_patt_q2.order - 2

            add_patt_q3.x_mm = add_patt_q3.x_mm + self.dx / 2
            add_patt_q3.y_mm = add_patt_q3.y_mm + 1. * self.dy / 3
            add_patt_q3.order = add_patt_q3.order + 2

            add_patt_q4.x_mm = add_patt_q4.x_mm - self.dx / 2
            add_patt_q4.y_mm = add_patt_q4.y_mm + 1. * self.dy / 3
            add_patt_q4.order = add_patt_q4.order - 2

            self.pattern = pd.concat([self.pattern, add_patt_q1, add_patt_q2, add_patt_q3, add_patt_q4])

        self.pattern = self.pattern.sort_values(by=['order'])

        # By construction, there might be some duplicates along x=0 or y=0. Those duplicates should be removed.
        # The following trick is done instead of drop_duplicates with multiple columns because of floating point precisions.
        # On a 32-bit system, the length of the area to be scanned must not exceed 4000 mm = 4 meter w.r.t. to the center of the scan.
        # The following three lines remove all duplicates which are closer than 0.1mm in x and y simultaneously.
        if not "custom" in self.geofile:
            self.pattern = self.pattern.assign(
                tmp=(1E5 * (self.pattern.x_mm * 1E1).round() + (self.pattern.y_mm * 1E1).round()).astype(int))
            self.pattern = self.pattern.drop_duplicates(subset=['tmp'], keep='first')
            self.pattern = self.pattern.drop(columns=["tmp"])

        self.pattern = self.pattern.assign(status=-1)
        self.pattern = self.pattern.assign(NAnnotations=0)

    def visualisePattern(self):
        self.openFigure()
        if self.matplot_ax != None:
            self.matplot_ax.cla()
        sizes = self.pattern.detail_level.map({0: 50, 1: 25, 2: 25})
        colors = self.pattern.status.map({-1: "b", 100: "magenta", 0: "g", 1: "r", 2: "orange"})
        self.matplot_ax.scatter(self.pattern.x_mm, self.pattern.y_mm, s=sizes, c=colors, zorder = 3)
        self.matplot_ax.grid(zorder = 1)
        self.matplot_ax.legend(handles=self.legend_elements, bbox_to_anchor=(1.01, 0.5))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.title(self.title)
        plt.pause(0.001)

    def initialise_scan(self, tmp_file_path=None):
        self.scan_sequence = sorted(self.pattern.order.unique())
        self.N_scan_points = len(self.scan_sequence)
        self.scan_step = 0
        self.start_time = time.time()

    def next_xy(self):
        self.scan_step += 1
        _exists, this_pad, this_x, this_y = self.scan_step_coordinates(self.scan_step)
        if not _exists:
            return -1
        else:
            self.showProgressBar()
            return (self.scan_step, this_pad, this_x, this_y)

    def scan_step_coordinates(self, _step):
        if _step > self.N_scan_points:
            print("Scan step", _step, "is not defined.")
            return (False, -1, -999, -999)
        else:
            scan_point = self.scan_sequence[_step - 1]
            this_step = self.pattern[self.pattern.order == scan_point]
            this_x = this_step.x_mm.iloc[0]
            this_y = this_step.y_mm.iloc[0]
            this_pad = this_step.padnr.iloc[0]
            self.pattern.loc[self.pattern.order == scan_point, "status"] = 100
            self.visualisePattern()
            return (True, this_pad, this_x, this_y)

    def showProgressBar(self):
        elapsed_time = time.time() - self.start_time
        remaining_time = -1
        if self.scan_step > 0:
            remaining_time = elapsed_time / self.scan_step * (self.N_scan_points - self.scan_step)
        print(f"Step", self.scan_step, " / ", self.N_scan_points,
              " Remaining (elapsed) time %.1f s (%.1f s)" % (remaining_time, elapsed_time), end="\r")

    def showProgressBarAI(self, scan_step, N_scan_points):
        elapsed_time = time.time() - self.start_time
        remaining_time = -1
        if scan_step > 0:
            remaining_time = elapsed_time / scan_step * (N_scan_points - scan_step)
        print(f"Step", scan_step, " / ", N_scan_points,
              " Remaining (elapsed) time %.1f s (%.1f s)" % (remaining_time, elapsed_time), end="\r")

    def setNAnnotations(self, scan_step, _n=0):
        if scan_step > self.N_scan_points:
            return
        scan_point = self.scan_sequence[scan_step - 1]
        if _n == 0:
            status = 0
        elif _n == 2:
            status = 2
        else:
            status = 1
        self.pattern.loc[self.pattern.order == scan_point, "NAnnotations"] = _n
        self.pattern.loc[self.pattern.order == scan_point, "status"] = status
        self.visualisePattern()


if __name__ == "__main__":
    sm = ScanMapping("custom_2patch_overlap")
    sm.loadGeoFile()
    sm.createPattern(detail_level=0)
    sm.visualisePattern()
    sm.initialise_scan()
    sm.next_xy()
    import pdb

    pdb.set_trace()

