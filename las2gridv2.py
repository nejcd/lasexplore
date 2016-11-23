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
import pandas as pd
import scipy
import datetime
from scipy.spatial.kdtree import KDTree
import matplotlib.pyplot as plt
from scipy import stats
import scipy.misc
import h5py

#Timer 1
t0 = datetime.datetime.now()

#Read data and set parameters
path = 's:/Dropbox/dev/Data/'
filename = '29'
las = laspy.file.File(path + filename + '.las', mode='r')
coords3d = np.vstack((las.x, las.y, las.z)).transpose()
values = np.vstack((las.classification, las.intensity)).transpose()
hf = h5py.File(path + filename + '_data.h5', 'w')

x = las.x
y = las.y
z = las.z

cutoff = 20

#Timer 2
time_delta_0 = datetime.datetime.now() - t0
print ('Time read and tree {0}'.format(time_delta_0))
t1 = datetime.datetime.now()

#Create grid and comput Stats
gridx = np.linspace(int(las.header.min[0]-cutoff), int(las.header.max[0]+cutoff),
                    int(las.header.max[0] - las.header.min[0] + 1 + 2 * cutoff))
gridy = np.linspace(int(las.header.min[1]-cutoff), int(las.header.max[1]+cutoff),
                    int(las.header.max[1] - las.header.min[1] + 1 + 2 * cutoff))

mean, _, _, _, = scipy.stats.binned_statistic_2d(x, y, z, statistic='mean', bins=[gridx, gridy])
zmin, _, _, _, = scipy.stats.binned_statistic_2d(x, y, z, statistic=lambda zmi: np.min(z), bins=[gridx, gridy])
zmax, _, _, _, = scipy.stats.binned_statistic_2d(x, y, z, statistic=lambda zma: np.max(z), bins=[gridx, gridy])

#Output overall img
f1 = 255 * scipy.special.expit(mean)
f2 = 255 * scipy.special.expit(zmin)
f3 = 255 * scipy.special.expit(zmax)

feature = np.array([f1, f2, f3])
scipy.misc.toimage(feature, cmin=0.0, cmax=255).save('feat_out\outfile.jpg')
time_delta_1 = datetime.datetime.now() - t1

#Timer 3
print ('For stats it took {0}'.format(time_delta_1))
t2 = datetime.datetime.now()
features = []
n = 0
for point, value in zip(coords3d, values):

    centerx = len(gridx[gridx < point[0]])
    centery = len(gridy[gridy < point[1]])

    f1 = 255 * scipy.special.expit(mean[(centerx - cutoff):(centerx + cutoff),
                                   (centery - cutoff):(centery + cutoff)] - point[2])
    f2 = 255 * scipy.special.expit(zmin[(centerx - cutoff):(centerx + cutoff),
                                   (centery - cutoff):(centery + cutoff)] - point[2])
    f3 = 255 * scipy.special.expit(zmax[(centerx - cutoff):(centerx + cutoff),
                                   (centery - cutoff):(centery + cutoff)] - point[2])

    feature = np.array([f1, f2, f3], dtype=np.uint8)

    features.append(feature)

    #scipy.misc.toimage(feature, cmin=0.0, cmax=255).save('feat_out\outfile{0}.jpg'.format(n))

    n += 1
    if n > 1000:
       break


#hf.create_dataset('features', data=features)
#hf.create_dataset('labels', data=las.classification)

#Timer stop
time_delta_2 = datetime.datetime.now() - t2
print ('For features it took {0}'.format(time_delta_2))
