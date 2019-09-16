"""
Checks the reconstruction error of an overall detection.
"""

import math
import numpy as np
import statistics
import matplotlib.pyplot as plt
from plot_baseband_RFI import plot_detection

errors = np.load('../data/errors.npy')
bbfiles = np.load('../data/labels.npy')

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



fig, (ax1, ax2) = plt.subplots(2,1)
ax1.hist(stddevs, 50, density=True)
ax1.set_title("STDEV of reconstruction error for each bbFile")

ax2.hist(avgs, 50, density=True)
ax2.set_title("AVG of reconstruction error for each bbFile")

plt.savefig("../figures/bbStats.png")

i = np.argmax(np.array(avgs))

plot_detection(bb_file=labels[i], error=avgs[i])