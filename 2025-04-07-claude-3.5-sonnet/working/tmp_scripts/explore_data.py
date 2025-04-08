"""
Initial exploration of the NWB file to understand:
1. Basic dataset statistics
2. Spike timing relative to stimulation trials
3. Distribution of spike rates across units
4. Basic visualization of spike raster around stimulation
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Set up seaborn style
sns.set_theme()

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000945/assets/e35653b4-0a0b-41bf-bf71-0c37e0d96509/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get trials info
trials = nwb.intervals["trials"]
trial_starts = trials["start_time"][:]
trial_stops = trials["stop_time"][:]
print(f"\nNumber of trials: {len(trial_starts)}")
print(f"Average trial duration: {np.mean(trial_stops - trial_starts):.3f} seconds")

# Get units info
units = nwb.units
unit_ids = units["id"].data
num_units = len(unit_ids)
print(f"Number of units: {num_units}")

# Calculate firing rates and collect spikes around trials
firing_rates = []
spikes_around_stim = defaultdict(list)
window = [-0.5, 1.0]  # Look at spikes from 0.5s before to 1s after stim

# Analyze first 10 trials and first 20 units for visualization
for unit_idx in range(20):
    spike_times = units["spike_times"][unit_idx]
    firing_rates.append(len(spike_times) / (trial_stops[-1] - trial_starts[0]))
    
    for trial_idx in range(10):
        trial_start = trial_starts[trial_idx]
        mask = (spike_times >= trial_start + window[0]) & (spike_times <= trial_start + window[1])
        relative_times = spike_times[mask] - trial_start
        spikes_around_stim[unit_idx].extend(relative_times)

# Plot 1: Firing rate distribution
plt.figure(figsize=(8, 6))
plt.hist(firing_rates, bins=20)
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Number of Units')
plt.title('Distribution of Firing Rates')
plt.savefig('tmp_scripts/firing_rates.png')
plt.close()

# Plot 2: Raster plot around stimulation
plt.figure(figsize=(10, 8))
for unit_idx in range(20):
    spikes = spikes_around_stim[unit_idx]
    plt.scatter(spikes, [unit_idx] * len(spikes), marker='|', s=10, color='black', alpha=0.5)
plt.axvline(x=0, color='r', linestyle='--', label='Stimulation')
plt.xlabel('Time relative to stimulation (s)')
plt.ylabel('Unit')
plt.title('Spike Raster Plot Around Stimulation\n(First 20 units, first 10 trials)')
plt.legend()
plt.savefig('tmp_scripts/raster_plot.png')
plt.close()

# Print some basic stats
print("\nFiring rate statistics:")
print(f"Mean firing rate: {np.mean(firing_rates):.2f} Hz")
print(f"Median firing rate: {np.median(firing_rates):.2f} Hz")
print(f"Min firing rate: {np.min(firing_rates):.2f} Hz")
print(f"Max firing rate: {np.max(firing_rates):.2f} Hz")