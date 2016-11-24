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

def createGrid(points, extend, training, buffer=20, values=[]):
    """Create grid and caluclate features
    :param points: Array Vstack [x, y, z, intensity, classification, id] [m]
    :type points: float
    :param extend: Array [minX minY, maxX maxY]
    :type extend: float
    :param training: If this is training data True else False ( training data should have label in 4th column
    :type training: bool
    :param buffer: Buffer area
    :type buffer: int
    :param values: Values for Feature Stats, if non is passed height is used
    :type training: float
    """
    x = points[:, 0]
    y = points[:, 1]

    if not values:
        values = points[:, 2]

    minX = extend[0, 0] + buffer
    minY = extend[0, 1] + buffer
    maxX = extend[1, 0] - buffer
    maxY = extend[1, 1] - buffer

    # Create grid and compute Stats

    gridx = np.linspace(int(minX - buffer), int(maxX + buffer),
                        int(maxX - minX + 1 + 2 * buffer))
    gridy = np.linspace(int(minY - buffer), int(maxY + buffer),
                        int(maxY - minY + 1 + 2 * buffer))

    mean, _, _, _, = scipy.stats.binned_statistic_2d(x, y, values, statistic='mean', bins=[gridx, gridy])
    zmin, _, _, _, = scipy.stats.binned_statistic_2d(x, y, values, statistic=lambda zmi: np.min(values), bins=[gridx, gridy])
    zmax, _, _, _, = scipy.stats.binned_statistic_2d(x, y, values, statistic=lambda zma: np.max(values), bins=[gridx, gridy])

    # Output overall img
    f1 = 255 * scipy.special.expit(mean)
    f2 = 255 * scipy.special.expit(zmin)
    f3 = 255 * scipy.special.expit(zmax)

    feature = np.array([f1, f2, f3])
    scipy.misc.toimage(feature, cmin=0.0, cmax=255).save('feat_out\outfile.jpg')

    features = []
    meta = []
    n = 0
    points = clip(points, extend)


    for point, value in zip(points, values):

        centerx = len(gridx[gridx < point[0]])
        centery = len(gridy[gridy < point[1]])

        f1 = 255 * scipy.special.expit(mean[(centerx - buffer):(centerx + buffer),
                                       (centery - buffer):(centery + buffer)] - value)
        f2 = 255 * scipy.special.expit(zmin[(centerx - buffer):(centerx + buffer),
                                       (centery - buffer):(centery + buffer)] - value)
        f3 = 255 * scipy.special.expit(zmax[(centerx - buffer):(centerx + buffer),
                                       (centery - buffer):(centery + buffer)] - value)

        feature = np.array([f1, f2, f3], dtype=np.uint8)
        scipy.misc.toimage(feature, cmin=0.0, cmax=255).save('feat_out\outfile.jpg')

        if training:
            meta.append([point[3], point[4]])
        elif not training:
            meta.append(point[3])

        features.append(feature)
        break

    return features, meta

def clip(coordinats, extend):
    """Clip point to extend
    :param coordinats: Array Vstack [x, y] [m]
    :type coords2d: float
    :param value: Array of values for feature calculation (z, intensity, linearitiy etc...)
    :type value: float
    :param extend: Array [minX minY, maxX maxY]
    :type extend: float
    :param label: By default create features for learning samples, if you pas Labels it creates training data
    :type label: int
    """
    xmin = extend[0, 0]
    ymin = extend[0, 1]
    xmax = extend[1, 0]
    ymax = extend[1, 1]

    coordinats = coordinats[coordinats[:, 0] < xmax]
    coordinats = coordinats[coordinats[:, 0] > xmin]
    coordinats = coordinats[coordinats[:, 1] < ymax]
    coordinats = coordinats[coordinats[:, 1] > ymin]

    return coordinats

def printFeatures(features):

    n = 0
    for feature in features:
        scipy.misc.toimage(feature, cmin=0.0, cmax=255).save('feat_out\outfile{0}.jpg'.format(n))
        n += 1
        if n > 100:
            break

#Timer 1
t0 = datetime.datetime.now()

#Read data and set parameters
path = 's:/Dropbox/dev/Data/'
filename = '01'
las = laspy.file.File(path + filename + '.las', mode='r')
pointsin = np.vstack((las.x, las.y, las.z, las.classification, range(0, len(las.x)))).transpose()

#hf = h5py.File(path + filename + '_data.h5', 'w')

#Timer 2
time_delta_0 = datetime.datetime.now() - t0
print ('Time read and tree {0}'.format(time_delta_0))
t1 = datetime.datetime.now()

#
dx = int(np.floor(las.header.max[0] - las.header.min[0]))
dy = int(np.floor(las.header.max[1] - las.header.min[1]))
sx = las.header.min[0]
sy = las.header.min[1]

step = 200
buffer = 20
features = []
meta = []
n = 0
if dx > step or dy > step:

    for x in range(0, dx/step):
        for y in range(0, dy/step):

            extend = np.array([[sx - buffer + (x * step), sy - buffer + (y * step)],
                               [sx + buffer + step + (x * step), sy + buffer + step + (y * step)]])

            coo = clip(pointsin, extend)

            if len(coo):
                f1, m1 = createGrid(coo, extend, True, buffer)
                features.append(f1)
                meta.append(m1)

            coo = []
            n += 1
            print 'Processing part {0} of {1}.'.format(n, x * y)

    extend = np.array([[sx - buffer + dx, sy - buffer + dy],
                       [sx + buffer + step + dx, sy + buffer + step + dy]])
    coo = clip(pointsin, extend)
    if len(coo):
        f1, m1 = createGrid(coo, extend, True, buffer)
        features.append(f1)
        meta.append(m1)

else:
    extend = np.array([[las.header.min[0] - buffer, las.header.min[1] - buffer],
                       [las.header.max[0] + buffer, las.header.max[1] + buffer]])

    f1, m1 = createGrid(pointsin, extend, True, buffer)
    features.append(f1)
    meta.append(m1)

printFeatures(features)

#Timer stop
time_delta_1 = datetime.datetime.now() - t1
print ('For features it took {0}'.format(time_delta_1))

#hf.create_dataset('features', data=features)
#hf.create_dataset('labels', data=las.classification)


