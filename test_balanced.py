import numpy as np

def downsample(points_length, sampling_rate):
    n = np.prod(points_length)
    x = np.fromstring(np.random.bytes(n), np.uint8, n)
    return (x < 255 * sampling_rate).reshape(points_length)

def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

labels = [1,2]
c = np.array([1,1,1,1,1,1,1,1,1,2,2,2,3,3,3,3,3])
x = np.ones(len(c))
y = np.ones(len(c))
z = np.ones(len(c))

pointsin = np.vstack((x, y, z, c)).transpose()
subsample = []

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

	if min_elements == number_elements:
		subsample.append(elements)
		print label
	elif number_elements > min_elements:
		subsample.append(elements[downsample(number_elements, float(min_elements)/number_elements)])

subsample = np.concatenate(subsample)
print subsample


