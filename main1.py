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
import datetime
import las2gridv5 as las2grid

#Read data and set parameters
path = '/media/nejc/Prostor/Dropbox/dev/Data/'
#path = 'e:/Dropbox/dev/Data/'

files = las2grid.get_list_of_las(path)
sampling_rate = 0.1

start = datetime.datetime.now()
files_to_go = len(files)
print ('Stating to process {0} files.'.format(files_to_go))
for file in files:
    print ('Processing file: {0}'.format(file))
    las = laspy.file.File(path + file, mode='r')
    pointsin = np.vstack((las.x, las.y, las.z, las.classification)).transpose()
    extend = las2grid.get_extend(las)
    #Timer 1

    features = las2grid.create_featureset(pointsin, extend, True, sampling_rate=sampling_rate)
    
    np.save(path + file + '.npy', features)

    #Timer 1
    time_delta = datetime.datetime.now() - start
    print ('File {0} finnisehd in {1}. To do: {2}'.format(file, time_delta, files_to_go))
    start = datetime.datetime.now()
    files_to_go -= 1