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
from scipy.spatial.kdtree import KDTree
import h5py


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

def create_featureset(points, extend, training, buffer=20,  values=[]):
    """Create grid and caluclate features
    :param points: Array Vstack [x, y, z, classification] [m]
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
    tree = KDTree(points[:, 0:2])

    if values:
        points[:, 2] = values


    features = []
    n = 0

    minX = extend[0, 0] - buffer
    minY = extend[0, 1] - buffer
    maxX = extend[1, 0] + buffer
    maxY = extend[1, 1] + buffer

    gridX = np.linspace(int(minX), int(maxX),
                        int(maxX - minX + 1))
    gridY = np.linspace(int(minY), int(maxY),
                        int(maxY - minY + 1))

    mean = np.zeros((len(gridX), len(gridY)))
    minm = np.zeros((len(gridX), len(gridY)))
    maxm = np.zeros((len(gridX), len(gridY)))

    for x, i in zip(gridX, range(0, len(gridX))):
        for y, j in zip(gridY, range(0, len(gridY))):
            list = tree.query_ball_point([x, y], 1.4)
            cell_ext = np.array([[x - 0.5, y - 0.5],
                               [x + 0.5, y + 0.5]])
            cell_points = clip(points[list], cell_ext)

            if cell_points.any():

                mean[i, j] = np.mean(cell_points[:, 2])
                minm[i, j] = np.min(cell_points[:, 2])
                maxm[i, j] = np.max(cell_points[:, 2])

    # Output overall img
    f1 = 255 * scipy.special.expit(mean)
    f2 = 255 * scipy.special.expit(minm)
    f3 = 255 * scipy.special.expit(maxm)

    feature = np.array([f1, f2, f3])
    scipy.misc.toimage(feature, cmin=0.0, cmax=255).save('feat_out\outfile.jpg')

    for point in points:

        n += 1
        
        centerx = len(gridX[gridX < point[0]])
        centery = len(gridY[gridY < point[1]])

        f1 = 255 * scipy.special.expit(mean[(centerx - buffer):(centerx + buffer),
                                        (centery - buffer):(centery + buffer)] - point[2])
        f2 = 255 * scipy.special.expit(minm[(centerx - buffer):(centerx + buffer),
                                        (centery - buffer):(centery + buffer)] - point[2])
        f3 = 255 * scipy.special.expit(maxm[(centerx - buffer):(centerx + buffer),
                                        (centery - buffer):(centery + buffer)] - point[2])

        feature = np.array([f1, f2, f3], dtype=np.uint8).reshape(40,40,3)
        if training:
            if int(point[3] != 2):
                features.append((feature, [0, 1]))
            elif int(point[3] == 2):
                features.append((feature, [1, 0]))

            #scipy.misc.toimage(feature, cmin=0.0, cmax=255).save('feat_out\outfile{0}.jpg'.format(n))


    return features

def printFeatures(features):

    n = 0
    for feature in features:
        scipy.misc.toimage(feature, cmin=0.0, cmax=255).save('feat_out\outfile{0}.jpg'.format(n))
        if n > 10:
            break


def save_npy(featureset):
    np.save(path + filename + '.npy', featureset)




########################################################
#            MAIN CODE                                 #
########################################################
if __name__ == '__main__':

    t0 = datetime.datetime.now()
    #Read data and set parameters
    path = '/media/nejc/Prostor/Dropbox/dev/Data/'
    #path = 'e:/Dropbox/dev/Data/'
    filename = '29'
    las = laspy.file.File(path + filename + '.las', mode='r')
    pointsin = np.vstack((las.x, las.y, las.z, las.classification)).transpose()

    buffer = 20 #Size of offsets from point of interest and buffer around area
    extend = np.array([[las.header.min[0], las.header.min[1]],
                    [las.header.max[0], las.header.max[1]]])

    #Timer 1
    time_delta_0 = datetime.datetime.now() - t0
    print ('Time read and tree {0}'.format(time_delta_0))
    t1 = datetime.datetime.now()
  

    features = create_featureset(pointsin, extend, True, buffer)
    #Timer 2
    time_delta_1 = datetime.datetime.now() - t1
    print ('For features it took {0}'.format(time_delta_1))
    t2 = datetime.datetime.now()

    
    time_delta_2 = datetime.datetime.now() - t2
    save_npy(features)
    print ('Time to save npy {0}'.format(time_delta_2))