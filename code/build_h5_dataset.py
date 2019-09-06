"""
Makes a h5 file containing a matrix of autoencoder inputs (not a tensor yet) as well as the labels
"""

import h5py
import numpy as np
import glob
from utils import round_down

segment_size=500

X = []
X_labels = np.array([])

# Grab all the waveforms
files = glob.glob("/media/nsbruce/Backup Plus/460MHz/waveforms/*")
    
for file in files[1000:2000]:
    data = np.fromfile(file)
    # Get length that rounds to nearest multiple of segment size
    new_len = round_down(len(data), segment_size)
    if new_len > 0:
        data = data[:new_len]
        i = 0
        while len(data)-i > 0:
            X.append(data[i:i+segment_size])
            i += segment_size
            X_labels = np.append(X_labels, str(file))
    print("File: {}, len(X): {}, len(X_labels): {}".format(file, len(X), len(X_labels)))

X = np.array(X)
print("File: {}, X.shape: {}, X_labels.shape: {}".format(file, X.shape, X_labels.shape))

np.save('reshapedwaveforms_second1000.npy', X)
np.save('reshapedwaveforms_second1000labels.npy', X_labels)