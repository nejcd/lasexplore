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
import os, glob

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

def create_featureset(points, extend, training, sampling_rate=1, img_size=32, values=[]):
    """Create grid and caluclate features
    :param points: Array Vstack [x, y, z, classification] [m]
    :type points: float
    :param extend: Array [minX minY, maxX maxY]
    :type extend: float
    :param training: If this is training data True else False ( training data should have label in 4th column
    :type training: bool
    :param sampling: Values 0-1. By deafult is 1 (all data poitns), for 10% of dataset 0.1
    :type sampling: float
    :param img_size: Spatial size of feature area Default 32. Should be 2 to power of n
    :type img_size: int
    :param values: Values for Feature Stats, if non is passed height is used
    :type values: float
    """
    tree = KDTree(points[:, 0:2])
    buff = int(img_size*0.5)

    if values:
        points[:, 2] = values

    features = []
    n = 0

    minX = extend[0, 0] - buff
    minY = extend[0, 1] - buff
    maxX = extend[1, 0] + buff
    maxY = extend[1, 1] + buff

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

    

    if sampling_rate != 1:
        orig_point_count = len(points)
        points = points[downsample(len(points), sampling_rate)]
        print ('Processing {0} procent of points ({1} of {2})'.format(sampling_rate*100,len(points),orig_point_count))
    else:
        print ('Processing all {0} points'.format(len(points)))


    for point in points:

        n += 1
        
        centerx = len(gridX[gridX < point[0]])
        centery = len(gridY[gridY < point[1]])

        f1 = scipy.special.expit(mean[(centerx - buff):(centerx + buff),
                                        (centery - buff):(centery + buff)] - point[2])
        f2 = scipy.special.expit(minm[(centerx - buff):(centerx + buff),
                                        (centery - buff):(centery + buff)] - point[2])
        f3 = scipy.special.expit(maxm[(centerx - buff):(centerx + buff),
                                        (centery - buff):(centery + buff)] - point[2])

        feature = np.array([f1, f2, f3], dtype=np.uint8).reshape(img_size,img_size,3)
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

def downsample(points_length, sampling_rate):
    n = np.prod(points_length)
    x = np.fromstring(np.random.bytes(n), np.uint8, n)
    return (x < 255 * sampling_rate).reshape(points_length)

def get_extend(las):
    e = np.array([[las.header.min[0], las.header.min[1]],
                 [las.header.max[0], las.header.max[1]]])
    return e

def get_list_of_las(directory):
    os.chdir(directory)
    return glob.glob("*.las")



########################################################
#            MAIN CODE                                 #
########################################################
if __name__ == '__main__':

    t0 = datetime.datetime.now()
    #Read data and set parameters
    path = '/media/nejc/Prostor/Dropbox/dev/Data/'
    #path = 'e:/Dropbox/dev/Data/'
    filename = '01'
    sampling_rate = 0.1
    las = laspy.file.File(path + filename + '.las', mode='r')
    pointsin = np.vstack((las.x, las.y, las.z, las.classification)).transpose()

    
    extend = np.array([[las.header.min[0], las.header.min[1]],
                    [las.header.max[0], las.header.max[1]]])

    #Timer 1
    time_delta_0 = datetime.datetime.now() - t0
    print ('Time read and tree {0}'.format(time_delta_0))
    t1 = datetime.datetime.now()
  

    features = create_featureset(pointsin, extend, True, sampling_rate=sampling_rate)
    #Timer 2
    time_delta_1 = datetime.datetime.now() - t1
    print ('For features it took {0}'.format(time_delta_1))
    t2 = datetime.datetime.now()

    
    time_delta_2 = datetime.datetime.now() - t2
    save_npy(features)
    print ('Time to save npy {0}'.format(time_delta_2))