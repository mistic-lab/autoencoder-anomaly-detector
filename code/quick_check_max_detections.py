from plot_baseband_RFI import plot_detection
import numpy as np

path = '/Volumes/Backup Plus/'
waveform_path = path+'waveforms/'
h5_path = path+'Spectra/'

avgs = np.load(path+'avgs.npy')
labels = np.load(path+'condensedlabels.npy')

bbFiles = [
    waveform_path+'1565292014.dat.139.c64',
    waveform_path+'1565292618.dat.39.c64',
    waveform_path+'1565291105.dat.129.c64',
    waveform_path+'1565293815.dat.53.c64',
    waveform_path+'1565294109.dat.86.c64',
    waveform_path+'1565290425.dat.31.c64'
]

indexes = [116, 197, 1107, 1674, 923, 457]

zipped = zip(bbFiles, indexes)

for file, idx in zipped:
    plot_detection(bb_file=file, error=avgs[idx])