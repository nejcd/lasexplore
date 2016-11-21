#!/usr/bin/env python
"""
/***************************************************************************
 Get Lidar Slovenia data.

                              -------------------
        begin                : 2016-11-12
        git sha              : $Format:%H$
        copyright            : (C) 2016 by Nejc Dougan
        email                : nejc.dougan@gmail.com
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import laspy, laspy.file
import numpy as np

import scipy
import datetime
from scipy.spatial.kdtree import KDTree
import matplotlib.pyplot as plt
from scipy import stats


t0 = datetime.datetime.now()
path = 'S:\Dropbox\dev\Data/'
filename = '01'

las = laspy.file.File(path + filename + '.las', mode='r')
coords = np.vstack((las.x, las.y, las.z)).transpose()
values = np.vstack((las.classification, las.intensity)).transpose()
tree = KDTree(coords)

time_delta_0 = datetime.datetime.now() - t0
print ('Time read and tree {0}'.format(time_delta_0))
z=las.z


features = []
n = 0
t1 = datetime.datetime.now()
for point, value in zip(coords, values):

    gridx = np.linspace(point[0] - 20, point[0] + 20, 40)
    gridy = np.linspace(point[1] - 20, point[1] + 20, 40)

    mean, _, _, _, = scipy.stats.binned_statistic_2d(las.x, las.y, las.z, statistic='mean', bins=[gridx, gridy])
    min, _, _, _, = scipy.stats.binned_statistic_2d(las.x, las.y, z, statistic=lambda z: np.min(z), bins=[gridx, gridy])
    #max, _, _, _, = scipy.stats.binned_statistic_2d(las.x, las.y, las.z, statistic='max', bins=[gridx, gridy])

    n += 1
    if n > 100:
        break


time_delta_1 = datetime.datetime.now() - t1



print ('For {0} it took {1}'.format(n, time_delta_1))