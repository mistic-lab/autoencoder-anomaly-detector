"""
Makes four npy files:
    -training/testing matrix of autoencoder inputs (not a tensor yet)
    -labels for them all
    -then a smaller validation matrix
    -label set for validation matrix
"""

import numpy as np
import glob
from utils import round_down

segment_size=500
fileCutoff = 4500

X = []
X_labels = np.array([])

# Grab all the waveforms
files = glob.glob("/home/nsbruce/RFI/data/waveforms/*")
    
for file in files[:fileCutoff]:
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
    # print("File: {}, len(X): {}, len(X_labels): {}".format(file, len(X), len(X_labels)))

X = np.array(X)
print("Training/testing | X.shape: {}, X_labels.shape: {}".format(X.shape, X_labels.shape))

np.save('/home/nsbruce/RFI/data/training_testing_waveforms.npy', X)
np.save('/home/nsbruce/RFI/data/training_testing_waveform_labels.npy', X_labels)

X = []
X_labels = np.array([])
    
for file in files[fileCutoff:]:
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
    # print("File: {}, len(X): {}, len(X_labels): {}".format(file, len(X), len(X_labels)))

X = np.array(X)
print("Validation | X.shape: {}, X_labels.shape: {}".format(X.shape, X_labels.shape))

np.save('/home/nsbruce/RFI/data/validation_waveforms.npy', X)
np.save('/home/nsbruce/RFI/data/validation_waveform_labels.npy', X_labels)
