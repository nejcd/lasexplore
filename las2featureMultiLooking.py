import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import datetime
import laspy, laspy.file
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



# Every 100 data samples, we save 1. If things run too
# slow, try increasing this number. If things run too fast,
# try decreasing it... =)
reduce_factor = 100


# Look pretty...
matplotlib.style.use('ggplot')


# Load up the scanned armadillo
path = '/media/nejc/Prostor/AI/data/test_arranged_class_labels/test/'
filename = '02'

las = laspy.file.File(path + filename + '.las', mode='r')
pointsin = np.vstack((las.x, las.y, las.z,)).transpose()


def do_PCA(pointsin):
  from sklearn.decomposition import PCA
  pca = PCA(n_components=2)
  pca.fit(pointsin)
  PCA(copy=True, n_components=2, whiten=False)
  
  T = pca.transform(pointsin)
  

  return T



# Render the Original
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_title('LSS 3D')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter(las.x, las.y, las.z, c='green', marker='.', alpha=0.75)



# Render the newly transformed PCA armadillo!
pca = do_PCA(pointsin)
if not pca is None:
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title('PCA')
  ax.scatter(pca[:,0], pca[:,1], c='blue', marker='.', alpha=0.75)


plt.show()

