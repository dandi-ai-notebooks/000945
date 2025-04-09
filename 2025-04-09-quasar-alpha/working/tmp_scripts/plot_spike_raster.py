"""
Generate a spike raster plot aligned to trials for a subset of units and trials.
Outputs raster image as tmp_scripts/spike_raster.png.
"""

import remfile
import h5py
import pynwb
import matplotlib.pyplot as plt
import numpy as np

# Load NWB file via recommended method
url = "https://api.dandiarchive.org/api/assets/f88a9bec-23d6-4444-8b97-8083e45057c9/download/"
file_obj = remfile.File(url)
f = h5py.File(file_obj)
io = pynwb.NWBHDF5IO(file=f)
nwbfile = io.read()

# Get trial info
trial_table = nwbfile.trials
num_trials = len(trial_table.id)
trial_start_times = trial_table['start_time'][:]
trial_stop_times = trial_table['stop_time'][:]

# Select first 20 trials for plotting
max_trials = 20
trial_idxs = np.arange(min(max_trials, num_trials))

# Get units info
units_table = nwbfile.units
num_units_total = len(units_table.id)
# Select first 30 units
max_units = 30
unit_idxs = np.arange(min(max_units, num_units_total))

plt.figure(figsize=(15, 8))

for j, unit_idx in enumerate(unit_idxs):
    spike_times = units_table['spike_times'][unit_idx]
    # Filter spikes: keep only those within selected trials
    mask = np.zeros_like(spike_times, dtype=bool)
    for trial_i in trial_idxs:
        mask |= (spike_times >= trial_start_times[trial_i]) & (spike_times <= trial_stop_times[trial_i])
    selected_spikes = spike_times[mask]
    plt.scatter(selected_spikes, np.full_like(selected_spikes, j), s=2, color='k')

# Draw vertical lines for trial boundaries
for trial_i in trial_idxs:
    plt.axvline(trial_start_times[trial_i], color='red', linestyle='--', alpha=0.3)
    plt.axvline(trial_stop_times[trial_i], color='red', linestyle='--', alpha=0.3)

plt.xlabel("Time (s)")
plt.ylabel("Units (subset)")
plt.title("Spike raster plot for subset of units and trials")
plt.tight_layout()
plt.savefig("tmp_scripts/spike_raster.png")