# Histogram of trial durations for Dandiset 000945 NWB file

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://api.dandiarchive.org/api/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/download/"

file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

trials = nwb.trials
start_times = trials['start_time'][:]
stop_times = trials['stop_time'][:]
durations = stop_times - start_times

sns.set_theme()

plt.figure(figsize=(8,5))
plt.hist(durations, bins=20, edgecolor='black')
plt.xlabel('Trial duration (s)')
plt.ylabel('Count')
plt.title('Distribution of trial durations')
plt.tight_layout()
plt.savefig('tmp_scripts/trial_durations.png')

io.close()