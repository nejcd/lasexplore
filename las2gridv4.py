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
from scipy import stats
import scipy.misc
import h5py

def createGrid(points, training, buffer=20, values=[]):
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

    if values:
        points[:, 2] = values

    features = []
    n = 0
    for point in points:

        x0 = int(point[0] - 0.5 - buffer)
        y0 = int(point[1] - 0.5 - buffer)

        gridx = np.linspace(int(x0), int(x0 + 2 * buffer),
                            int(1 + 2 * buffer))
        gridy = np.linspace(int(y0), int(y0 + 2 * buffer),
                            int(1 + 2 * buffer))

        extend = np.array([[x0, y0],
                           [x0 + 1 + 2 * buffer, y0 + 1 + 2 * buffer]])

        clippoint = clip(points, extend)

        x = clippoint[:, 0]
        y = clippoint[:, 1]
        values = clippoint[:, 2]

        mean, _, _, _, = scipy.stats.binned_statistic_2d(x, y, values, statistic='mean', bins=[gridx, gridy])
        zmin, _, _, _, = scipy.stats.binned_statistic_2d(x, y, values, statistic=lambda zmi: np.min(values),
                                                         bins=[gridx, gridy])
        zmax, _, _, _, = scipy.stats.binned_statistic_2d(x, y, values, statistic=lambda zma: np.max(values),
                                                         bins=[gridx, gridy])

        f1 = 255 * scipy.special.expit(mean - point[2])
        f2 = 255 * scipy.special.expit(zmin - point[2])
        f3 = 255 * scipy.special.expit(zmax - point[2])
        feature = np.array([f1, f2, f3], dtype=np.uint8)

        features.append(feature)

        n += 1
        # if n < 100:
        #     scipy.misc.toimage(feature, cmin=0.0, cmax=255).save('feat_out\outfile{0}.jpg'.format(n))
        # else:
        #     break

    return features

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
        if n > 100:
            break

########################################################
#            MAIN CODE                                 #
########################################################

#Timer 1
t0 = datetime.datetime.now()

#Read data and set parameters
path = 's:/Dropbox/dev/Data/'
filename = '01'
las = laspy.file.File(path + filename + '.las', mode='r')
pointsin = np.vstack((las.x, las.y, las.z, las.classification)).transpose()
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

buffer = 20

features = createGrid(pointsin, True, buffer)
#printFeatures(features)

#Timer stop
time_delta_1 = datetime.datetime.now() - t1
print ('For features it took {0}'.format(time_delta_1))

#hf.create_dataset('features', data=features)
#hf.create_dataset('labels', data=las.classification)


