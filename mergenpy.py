#!/usr/bin/env python
"""
/***************************************************************************
 

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

import numpy as np
import os, glob


#path = '/media/nejc/Prostor/AI/data/kelag_32_MSS/'
path = '/media/nejc/Prostor/AI/data/test_arranged_class_labels/class_2_64x64_balanced_MMM_kelag_all/'

os.chdir(path)
files = glob.glob("*.npy")

files_to_go = len(files)
print ('Starting to process {0} files.'.format(files_to_go))

merged = []
n = 0
for filename in files:
	print "Status: {0} %".format((n))
	features = np.load(path + filename)
	merged.append(features)
	n += 1

merged = np.concatenate(merged)
np.random.shuffle(merged)

np.save(path + filename + '_merged.npy', merged)