"""
Checks the reconstruction error of an overall detection.
"""

import math
import numpy as np
import statistics
import matplotlib.pyplot as plt

errors = np.load('errors.npy')
bbfiles = np.load('bbfiles.npy')

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

for value in collectionDict.values():
    stddevs.append(np.std(value))



plt.hist(stddevs, 50, density=True)
plt.title("STDEV of reconstruction error for each bbFile")
plt.show()
