"""
Script wrapper around plot_baseband_RFI.py
"""

import numpy as np
import math
import h5py
import argparse
import os
import matplotlib.pyplot as plt
from plot_baseband_RFI import plot_detection

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


    
plot_detection(args.bb_file, args.h5, args.index, args.error)