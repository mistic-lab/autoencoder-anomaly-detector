"""
Takes a baseband file or full h5 file and detection index and plots the passband waterfall of it.
"""

import numpy as np
import math
import h5py
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    description='Plots passband waterfall from baseband reference', 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
parser.add_argument('--bb-file', type=str,
                    help=".c64 baseband file to plot")
parser.add_argument('--h5', type=str,
                    help="h5 file containing the full experiment")
parser.add_argument('--index', type=str,
                    help="merged_detection index to plot")
parser.add_argument("--error", type=float,
                    help='reconstruction error for title only')
args = parser.parse_args()

if args.bb_file is None:
    if args.h5 is None or args.index is None:
        raise Exception("No BB file was specified, so you must provide both an h5 file and a detection index.")
    else:
        idx = args.index
        h5File = args.h5
else:
    if args.h5 is not None or args.index is not None:
        print('A BB File was specified, all other flags (--h5 and --index) are being ignored.')
    path, bb_file = os.path.split(args.bb_file)
    print(bb_file)
    parentName, dat, idx, c64 = bb_file.split('.')
    idx = int(idx)
    path = path.split('waveforms')[0]
    h5File = path+parentName+'.h5'

if args.error is None:
    title='File: {}, RFI index {}'.format(parentName+'.h5', idx)
else:
    title=title='File: {}, RFI index {}, Reconstruction error {}'.format(parentName+'.h5', idx, str(args.error))

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
    plt.show()