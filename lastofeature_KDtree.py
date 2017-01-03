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
import matplotlib
import scipy
import datetime
from scipy.spatial.kdtree import KDTree
from scipy import special



t0 = datetime.datetime.now()
path = '/media/nejc/Prostor/AI/data/test_arranged_class_labels/test/'
filename = '02'

las = laspy.file.File(path + filename + '.las', mode='r')
coords = np.vstack((las.x, las.y, las.z)).transpose()
values = np.vstack((las.classification, las.intensity)).transpose()
tree = KDTree(coords)

time_delta_0 = datetime.datetime.now() - t0
print ('Time read and tree {0}'.format(time_delta_0))

features = []
n = 0
t1 = datetime.datetime.now()
for point, value in zip(coords, values):
    print point
    [dist, i] = tree.query(point, k=100)
    #keep = dist < 2
    [lambda_3, lambda_2, lambda_1] = scipy.linalg.eigh(np.cov(coords[i].transpose()), eigvals_only=True)

    # Features
    linearity = (lambda_1 - lambda_2) / lambda_1
    planarity = (lambda_2 - lambda_3) / lambda_1
    scattering = lambda_3 / lambda_1
    omnivariance = scipy.special.cbrt(lambda_1 * lambda_2 * lambda_3)
    anisotropy = (lambda_1 - lambda_3) / lambda_1
    eigentropy = -(lambda_1 * np.log(lambda_1)
                   + lambda_2 * np.log(lambda_2)
                   + lambda_3 * np.log(lambda_3))
    curvature = lambda_3 / (lambda_1 + lambda_2 + lambda_3)

    features.append([linearity, planarity, scattering, omnivariance, anisotropy, eigentropy, value[0], value[1]])
    n += 1
    #if n > 100:
    #    break


#np.savetxt(path + filename + '.csv', features, delimiter=",")
#np.save(path + filename + '.npy', features)

time_delta_1 = datetime.datetime.now() - t1



print ('For {0} it took {1}'.format(n, time_delta_1))