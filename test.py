#!/usr/bin/env python


import laspy, laspy.file
import numpy as np
import scipy
import datetime
from scipy import stats
import scipy.misc
from scipy.spatial.kdtree import KDTree
import h5py
import las2gridv5 as las2grid

t0 = datetime.datetime.now()
#Read data and set parameters
path = '/media/nejc/Prostor/Dropbox/dev/Data/'
#path = 'e:/Dropbox/dev/Data/'

print las2grid.get_list_of_las(path)

#filename = '01'
#las = laspy.file.File(path + filename + '.las', mode='r')
#points = np.vstack((las.x, las.y, las.z, las.classification)).transpose()
