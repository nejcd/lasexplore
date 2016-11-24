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

path = 's:/Dropbox/dev/Data/'
filename = '01'
las = laspy.file.File(path + filename + '.las', mode='r')
pointsin = np.vstack((las.x, las.y, las.z, las.classification, range(0, len(las.x)))).transpose()

print len(pointsin)
print pointsin