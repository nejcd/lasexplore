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

t0 = datetime.datetime.now()
path = 'E:\Dropbox\dev\Data/'
filename = '01'

las = laspy.file.File(path + filename + '.las', mode='r')
coords3d = np.vstack((las.x, las.y, las.z)).transpose()
coords2d = np.vstack((las.x, las.y)).transpose()
values = np.vstack((las.classification, las.intensity)).transpose()
tree = KDTree(coords2d)

time_delta_0 = datetime.datetime.now() - t0
print ('Time read and tree {0}'.format(time_delta_0))

df = pd.DataFrame()
features = []
n = 0
t1 = datetime.datetime.now()
for point, value in zip(coords3d, values):

    gridx = np.linspace(point[0] - 20, point[0] + 20, 40)
    gridy = np.linspace(point[1] - 20, point[1] + 20, 40)

    list = tree.query_ball_point([point[0], point[1]], 56, p=2, eps=0)

    coo = coords3d[list]

    x = coo[:, 0]
    y = coo[:, 1]
    z = coo[:, 2]

    mean, _, _, _, = scipy.stats.binned_statistic_2d(x, y, z, statistic='mean', bins=[gridx, gridy])
    zmin, _, _, _, = scipy.stats.binned_statistic_2d(x, y, z, statistic=lambda zmi: np.min(z), bins=[gridx, gridy])
    zmax, _, _, _, = scipy.stats.binned_statistic_2d(x, y, z, statistic=lambda zma: np.max(z), bins=[gridx, gridy])

    f1 = 255*scipy.special.expit(mean - point[2])
    f2 = 255*scipy.special.expit(zmin - point[2])
    f3 = 255*scipy.special.expit(zmax - point[2])

    feature = np.array([f1, f2, f3])

    
    #df2 = pd.DataFrame(feature.reshape(-1,3))
    #df.append(df2)
    scipy.misc.toimage(feature, cmin=0.0, cmax=255).save('feat_out\outfile{0}.jpg'.format(n))

    n += 1
    if n > 100:
        break


time_delta_1 = datetime.datetime.now() - t1



print ('For {0} it took {1}'.format(n, time_delta_1))