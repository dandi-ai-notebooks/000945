# Plot spike counts per neuron/unit in the NWB file

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/download/"

file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

unit_ids = nwb.units.id[:]
counts = []
for idx in range(len(unit_ids)):
    spikes = nwb.units['spike_times'][idx]
    counts.append(len(spikes))

import seaborn as sns
sns.set_theme()

plt.figure(figsize=(8,5))
plt.hist(counts, bins=20, edgecolor='black')
plt.xlabel('Spike count per unit')
plt.ylabel('Number of units')
plt.title('Distribution of spike counts across neurons')
plt.tight_layout()
plt.savefig('tmp_scripts/spike_counts_per_unit.png')

io.close()