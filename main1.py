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
import las2featureVector as las2feature

#Read data and set parameters
path = '/media/nejc/Prostor/AI/data/test_arranged_class_labels/'
#path = 'e:/Dropbox/dev/Data/'

files = las2feature.get_list_of_las(path)
sampling_rate = 1

img_size = 32

start = datetime.datetime.now()
files_to_go = len(files)
print ('Starting to process {0} files.'.format(files_to_go))
for file in files:
    print ('Processing file: {0}'.format(file))
    labels_in = [5, 6]
    las = laspy.file.File(path + file, mode='r')
    pointsin = np.vstack((las.x, las.y, las.z, las.classification)).transpose()
    extend = las2feature.get_extend(las)
    #Timer 1

    #features = las2feature.create_featureset(pointsin, extend, labels_in,
    #                                         sampling_rate=sampling_rate,
    #                                         balanced=True,
    #                                         img_size=img_size)
    
    features = las2feature.create_feature(pointsin, labels_in, sampling_rate=1, balanced=True)

    np.save(path + file + '.npy', features)

    #Timer 1
    time_delta = datetime.datetime.now() - start
    print ('File {0} finnisehd in {1}. To do: {2}'.format(file, time_delta, files_to_go))
    start = datetime.datetime.now()
    files_to_go -= 1