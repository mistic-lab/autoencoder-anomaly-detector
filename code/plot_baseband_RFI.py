"""
Takes a baseband file or full h5 file and detection index and plots the passband waterfall of it.
"""

import numpy as np
import math
import h5py
import os
import matplotlib.pyplot as plt


def parse_bb_path(bb_file):
    path, bb_file = os.path.split(bb_file)
    print(bb_file)
    parentName, dat, idx, c64 = bb_file.split('.')
    idx = int(idx)
    path = path.split('waveforms')[0]
    h5File = path+'Spectra/'+parentName+'.h5'
    return idx, h5File

    
def plot_detection(bb_file=None, h5=None, index=None, error=None):
    if bb_file is None:
        if h5 is None or index is None:
            raise Exception("No BB file was specified, so you must provide both an h5 file and a detection index.")
        else:
            idx = index
            h5File = h5
    else:
        if h5 is not None or index is not None:
            print('A BB File was specified, all other flags (--h5 and --index) are being ignored.')
        idx, h5File = parse_bb_path(bb_file)
    
    if error is None:
        title='File: {}, RFI index {}'.format(h5File, idx)
    else:
        title='File: {}, Detection {}, Reconstruction error: {}%'.format(h5File[-13:], idx, str(error*100)[:4])


    with h5py.File(h5File, 'r') as f:
        
        margin = 5

        fig, ax = plt.subplots()

        x1, y1, x2, y2 = np.round(f['merged_detections'][:,idx])

        x1 = int(np.clip(x1-margin, 0, f['times'].shape[0]-1))
        x2 = int(np.clip(x2+margin, 0, f['times'].shape[0]-1))
        y1 = int(np.clip(y1-margin, 0, f['freqs'].shape[0]-1))
        y2 = int(np.clip(y2+margin, 0, f['freqs'].shape[0]-1))

        t1 = f['times'][x1]
        t2 = f['times'][x2]
        f1 = f['freqs'][y1]
        f2 = f['freqs'][y2]

        extents = [t1, t2, f2/1e6, f1/1e6]
        t1 = math.floor(t1)
        t2 = math.ceil(t2)
        f1 = math.floor(f1)
        f2 = math.ceil(f2)

        rfi = 10.*np.log10(f['psd'][y1:y2,x1:x2])
        im = ax.imshow(rfi, extent=extents, aspect='auto', interpolation='none')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Freq (MHz)')
        fig.colorbar(im, label='dB', ax=ax, orientation='vertical')
        ax.set_title(title)
        plt.show(block=False)




