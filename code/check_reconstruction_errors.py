"""
Checks the reconstruction error of an overall detection.
"""

import math
import numpy as np
import statistics
import matplotlib.pyplot as plt
from plot_baseband_RFI import plot_detection

errors = np.load('../data/errors.npy')
bbfiles = np.load('../data/bbfiles.npy')

collectionDict = {}
for i in range(len(bbfiles)):
    try:
        # adding to list in dict
        collectionDict[bbfiles[i]].append(errors[i])
    except KeyError:
        # create dict key and value list with item
        collectionDict[bbfiles[i]] = [errors[i]]


print("Number of bb files: {}".format(len(collectionDict)))
stddevs = []
avgs = []
labels = []

for key, value in collectionDict.items():
    stddevs.append(np.std(value))
    avgs.append(np.average(value))
    labels.append(key)




plt.hist(stddevs, 50, density=True)
plt.title("STDEV of reconstruction error for each bbFile")
plt.show()

plt.hist(avgs, 50, density=True)
plt.title("AVG of reconstruction error for each bbFile")
plt.show()

i = np.argmax(np.array(avgs))

plot_detection(bb_file=labels[i], error=avgs[i])