#!/usr/bin/env python


import laspy, laspy.file
import numpy as np
import scipy.misc


size = 32






from scipy import misc
f = misc.face()
misc.imsave('face.png', f) # uses the Image module (PIL)

import matplotlib.pyplot as plt
plt.imshow(f)
plt.show()

f1 = f[:,:,0]
f2 = f[:,:,1]
f3 = f[:,:,2]

rgbArray = np.empty((768, 1024, 3), 'uint8')
rgbArray[..., 0] = f1
rgbArray[..., 1] = f2
rgbArray[..., 2] = f3




scipy.misc.toimage(rgbArray, cmin=0.0, cmax=255).save('face2.jpg')


 # uses the Image module (PIL)

import matplotlib.pyplot as plt
plt.imshow(rgbArray)
plt.show()