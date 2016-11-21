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
import matplotlib.pyplot as plt
import matplotlib
from sklearn.decomposition import PCA
import scipy as sp
import datetime

# Look pretty...
matplotlib.style.use('ggplot')
from mpl_toolkits.mplot3d import Axes3D


#
t0 = datetime.datetime.now()

las = laspy.file.File('data/01.las', mode = 'r')
coords = np.vstack((las.x, las.y, las.z)).transpose()

time_delta_0 = datetime.datetime.now() - t0
print ('Time read and stack {0}'.format(time_delta_0))

n = 0
t1 = datetime.datetime.now()
for point in coords:
    distances = np.sum((coords - point)**2, axis = 1)
    keep = distances < 2

    [lambda_1, lambda_2, lambda_3] = sp.linalg.eigh(np.cov(coords[keep].transpose()), eigvals_only=True)

    #Features

    linearity = (lambda_1 - lambda_2) / lambda_1
    planarity = (lambda_2 - lambda_3) / lambda_1
    scattering = lambda_3 / lambda_1
    omnivariance = sp.special.cbrt(lambda_1 * lambda_2 * lambda_3)
    anisotropy = (lambda_1 - lambda_3) / lambda_1
    eigentropy = -(lambda_1 * np.log(lambda_1)
                   + lambda_2 * np.log(lambda_2)
                   + lambda_3 * np.log(lambda_3))
    curvature = lambda_3 / (lambda_1 + lambda_2 + lambda_3)

    n += 1
    if n > 100:
       break


time_delta = datetime.datetime.now() - t1



print ('For {0} it took {1}'.format(n, time_delta))