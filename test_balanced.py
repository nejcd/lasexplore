import numpy as np
import laspy, laspy.file
import datetime

def downsample(points_length, sampling_rate):
    n = np.prod(points_length)
    x = np.fromstring(np.random.bytes(n), np.uint8, n)
    return (x < 255 * sampling_rate).reshape(points_length)

def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

path = '/media/nejc/Prostor/AI/data/'
filename = '01'
labels = [5, 6]
las = laspy.file.File(path + filename + '.las', mode='r')
pointsin = np.vstack((las.x, las.y, las.z, las.classification)).transpose()


subsample = []
t0 = datetime.datetime.now()

all_labels = np.unique(pointsin)
other_labels = diff(all_labels, labels)
if not other_labels: labels.append(0)
for other_label in other_labels: pointsin[pointsin[:,3] == other_label] = 0

min_elements = None
for label in labels:
	number_elements = len(pointsin[pointsin[:,3] == label])
	if min_elements == None or min_elements > number_elements :
		min_elements = number_elements

for label in labels:
	elements = pointsin[pointsin[:,3] == label]
	number_elements = len(elements)
	if number_elements == min_elements:
		subsample.append(elements)
	elif number_elements > min_elements:
		subsample.append(elements[downsample(number_elements, float(min_elements)/number_elements)])

subsample = np.concatenate(subsample)
np.random.shuffle(subsample)

time_delta_0 = datetime.datetime.now() - t0
print ('Time to balance {0} dataset: {1}'.format(len(pointsin),time_delta_0))
print subsample
