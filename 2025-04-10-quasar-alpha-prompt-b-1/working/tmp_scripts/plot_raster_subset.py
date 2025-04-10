# Generate raster plot for a random subset of 10 units over a short initial time window

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns

url = "https://api.dandiarchive.org/api/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/download/"

file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

unit_ids = nwb.units.id[:]
n_units = len(unit_ids)
subset_size = min(10, n_units)
subset_indices = random.sample(range(n_units), subset_size)

plt.figure(figsize=(10, 6))
sns.set_theme()

time_window = 5.0  # seconds, visualize spikes in first 5 seconds

for j, idx in enumerate(subset_indices):
    spikes = nwb.units['spike_times'][idx]
    spikes_in_window = spikes[spikes < time_window]
    plt.vlines(spikes_in_window, j + 0.5, j + 1.5)

plt.xlabel('Time (s)')
plt.ylabel('Unit #')
plt.yticks(np.arange(1, subset_size + 1), labels=[str(unit_ids[idx]) for idx in subset_indices])
plt.title(f'Spike raster for {subset_size} randomly selected units\n(first {time_window}s)')
plt.tight_layout()
plt.savefig('tmp_scripts/raster_subset_units.png')

io.close()