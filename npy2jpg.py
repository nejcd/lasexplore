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
import scipy.misc

#path = '/media/nejc/Prostor/AI/data/kelag_32_MSS/'
path = '/media/nejc/Prostor/Dropbox/dev/Data/'
filename = '01'
every = 1000

features = np.load(path + filename + '.npy')
imgs = list(features[:,0])


n = 0
for img in imgs:
	n += 1

	if n % every == 0:
		img1 = img
		scipy.misc.toimage(img, cmin=0.0, cmax=255).save('{0}{1}_{2}.jpg'.format(path, filename, n))
