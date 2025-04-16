"""
This script explores neural activity in relation to trials of transcranial focused ultrasound stimulation.
It analyzes spike times of neurons before, during, and after stimulation to quantify the neural response.
"""

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/b88188c8-4e4c-494c-8dab-806b1efd55eb/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get trials dataframe
trials_df = nwb.trials.to_dataframe()
trial_starts = trials_df['start_time'].values
trial_stops = trials_df['stop_time'].values

# Get units dataframe
units_df = nwb.units.to_dataframe()

# Define time windows for analysis
pre_window = 1.0  # 1 second before stimulus
post_window = 1.0  # 1 second after stimulus

# Function to compute peri-stimulus time histogram (PSTH)
def compute_psth(spike_times, trial_starts, pre_window, post_window, bin_size=0.05):
    """Compute peri-stimulus time histogram around trial starts"""
    # Create time bins relative to stimulus onset
    bins = np.arange(-pre_window, post_window + bin_size, bin_size)
    bin_centers = bins[:-1] + bin_size/2
    
    # Count spikes in each bin for each trial
    counts = np.zeros((len(trial_starts), len(bins)-1))
    
    for i, start in enumerate(trial_starts):
        # Get spike times in window around trial start
        window_spikes = spike_times[(spike_times >= start - pre_window) & 
                                    (spike_times <= start + post_window)]
        # Convert to time relative to trial start
        relative_times = window_spikes - start
        # Count spikes in bins
        counts[i], _ = np.histogram(relative_times, bins=bins)
    
    # Average across trials and convert to firing rate
    mean_counts = np.mean(counts, axis=0)
    firing_rate = mean_counts / bin_size  # spikes per second
    
    return bin_centers, firing_rate

# Split units by cell type
rsu_units = units_df[units_df['celltype_label'] == 1]
fsu_units = units_df[units_df['celltype_label'] == 2]

print(f"Total number of units: {len(units_df)}")
print(f"Number of RSU (Regular Spiking Units): {len(rsu_units)}")
print(f"Number of FSU (Fast Spiking Units): {len(fsu_units)}")

# Analyze a subset of neurons (first 5 of each type)
rsu_subset = rsu_units.head(5)
fsu_subset = fsu_units.head(5)

# Compute average firing rates across all trials for each unit type
bin_size = 0.05  # 50 ms bins

# RSU firing rates
plt.figure(figsize=(10, 6))
for i, (idx, unit) in enumerate(rsu_subset.iterrows()):
    spike_times = unit['spike_times']
    bin_centers, firing_rate = compute_psth(spike_times, trial_starts, pre_window, post_window, bin_size)
    plt.plot(bin_centers, firing_rate, label=f"RSU Unit {idx}")

plt.axvline(x=0, color='k', linestyle='--', label='Stimulus Onset')
plt.axvline(x=2.2, color='r', linestyle='--', label='Stimulus Offset')
plt.xlabel('Time relative to trial start (s)')
plt.ylabel('Firing Rate (spikes/s)')
plt.title('RSU Firing Rates Around Stimulus')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('explore/rsu_firing_rates.png', dpi=300, bbox_inches='tight')
plt.close()

# FSU firing rates
plt.figure(figsize=(10, 6))
for i, (idx, unit) in enumerate(fsu_subset.iterrows()):
    spike_times = unit['spike_times']
    bin_centers, firing_rate = compute_psth(spike_times, trial_starts, pre_window, post_window, bin_size)
    plt.plot(bin_centers, firing_rate, label=f"FSU Unit {idx}")

plt.axvline(x=0, color='k', linestyle='--', label='Stimulus Onset')
plt.axvline(x=2.2, color='r', linestyle='--', label='Stimulus Offset')
plt.xlabel('Time relative to trial start (s)')
plt.ylabel('Firing Rate (spikes/s)')
plt.title('FSU Firing Rates Around Stimulus')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('explore/fsu_firing_rates.png', dpi=300, bbox_inches='tight')
plt.close()

# Compare average response between RSU and FSU
# Compute average across all RSUs and FSUs
rsu_avg_rates = []
fsu_avg_rates = []

# Use larger sample for more robust average
for i, (idx, unit) in enumerate(rsu_units.iterrows()):
    if i >= 15:  # Limit to 15 units to save processing time
        break
    spike_times = unit['spike_times']
    bin_centers, firing_rate = compute_psth(spike_times, trial_starts, pre_window, post_window, bin_size)
    rsu_avg_rates.append(firing_rate)

for i, (idx, unit) in enumerate(fsu_units.iterrows()):
    if i >= 15:  # Limit to 15 units
        break
    spike_times = unit['spike_times']
    bin_centers, firing_rate = compute_psth(spike_times, trial_starts, pre_window, post_window, bin_size)
    fsu_avg_rates.append(firing_rate)

rsu_mean = np.mean(rsu_avg_rates, axis=0) if rsu_avg_rates else np.zeros_like(bin_centers)
fsu_mean = np.mean(fsu_avg_rates, axis=0) if fsu_avg_rates else np.zeros_like(bin_centers)

plt.figure(figsize=(10, 6))
plt.plot(bin_centers, rsu_mean, label='RSU Mean', linewidth=2)
plt.plot(bin_centers, fsu_mean, label='FSU Mean', linewidth=2)
plt.axvline(x=0, color='k', linestyle='--', label='Stimulus Onset')
plt.axvline(x=2.2, color='r', linestyle='--', label='Stimulus Offset')
plt.xlabel('Time relative to trial start (s)')
plt.ylabel('Average Firing Rate (spikes/s)')
plt.title('Comparison of Average RSU and FSU Responses')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('explore/cell_type_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Clean up
io.close()
h5_file.close()
remote_file.close()

print("Plots saved to explore directory.")