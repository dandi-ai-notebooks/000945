"""
This script analyzes neural responses to transcranial focused ultrasound (tFUS) stimulation.
We want to understand:
1. How neurons respond to the stimulation
2. If there are differences in responses between cell types
3. If some neurons show stronger responses than others
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# Load
url = "https://api.dandiarchive.org/api/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get trial start times and units data
trials_df = nwb.trials.to_dataframe()
units_df = nwb.units.to_dataframe()

print(f"Analyzing responses for {len(units_df)} units across {len(trials_df)} trials")

# Define parameters for PSTH
pre_time = 1.0  # Time before stimulus onset (seconds)
post_time = 2.2  # Time after stimulus onset (seconds)
bin_size = 0.05  # Bin size for PSTH (seconds)
n_bins = int((pre_time + post_time) / bin_size)
time_bins = np.linspace(-pre_time, post_time, n_bins + 1)
time_centers = (time_bins[:-1] + time_bins[1:]) / 2

# Function to compute PSTH for one unit
def compute_psth(spike_times, trial_times, pre_time, post_time, bin_size):
    n_trials = len(trial_times)
    n_bins = int((pre_time + post_time) / bin_size)
    psth = np.zeros((n_trials, n_bins))
    
    for i, trial_start in enumerate(trial_times):
        # Find spikes in the window around this trial
        window_start = trial_start - pre_time
        window_end = trial_start + post_time
        
        # Convert to trial-relative time
        trial_spikes = spike_times[(spike_times >= window_start) & (spike_times < window_end)] - trial_start
        
        # Bin the spikes
        hist, _ = np.histogram(trial_spikes, bins=np.linspace(-pre_time, post_time, n_bins + 1))
        psth[i, :] = hist
    
    # Average across trials and convert to firing rate
    mean_psth = np.mean(psth, axis=0) / bin_size  # Convert to spikes/second
    sem_psth = stats.sem(psth, axis=0) / bin_size
    
    return mean_psth, sem_psth

# Extract trial start times
trial_starts = trials_df['start_time'].values

# Sample a subset of units to analyze (15 units - to avoid timeout)
np.random.seed(42)  # For reproducibility
unit_indices = np.random.choice(len(units_df), size=15, replace=False)

# Compute baseline and response firing rates for statistical comparison
baseline_window = (-0.9, -0.1)  # 0.8s window before stimulus
response_window = (0.1, 0.9)    # 0.8s window during stimulus

# Arrays to store results
unit_ids = []
cell_types = []
baseline_rates = []
response_rates = []
p_values = []
mean_psths = []

# Create a figure for PSTHs
plt.figure(figsize=(15, 10))

# Process each unit
for i, unit_idx in enumerate(unit_indices):
    unit_id = units_df.index[unit_idx]
    unit_ids.append(unit_id)
    
    # Get spike times and cell type
    spike_times = units_df.loc[unit_id, 'spike_times']
    cell_type = units_df.loc[unit_id, 'celltype_label']
    cell_types.append(cell_type)
    
    # Compute PSTH
    mean_psth, sem_psth = compute_psth(spike_times, trial_starts, pre_time, post_time, bin_size)
    mean_psths.append(mean_psth)
    
    # Calculate baseline and response firing rates for statistical test
    baseline_start_bin = int((baseline_window[0] + pre_time) / bin_size)
    baseline_end_bin = int((baseline_window[1] + pre_time) / bin_size)
    response_start_bin = int((response_window[0] + pre_time) / bin_size)
    response_end_bin = int((response_window[1] + pre_time) / bin_size)
    
    baseline_rate = np.mean(mean_psth[baseline_start_bin:baseline_end_bin])
    response_rate = np.mean(mean_psth[response_start_bin:response_end_bin])
    
    baseline_rates.append(baseline_rate)
    response_rates.append(response_rate)
    
    # Perform statistical test (paired t-test across trials)
    trial_baseline_rates = []
    trial_response_rates = []
    
    for trial_start in trial_starts:
        # Find spikes in baseline window
        baseline_spikes = spike_times[(spike_times >= trial_start + baseline_window[0]) & 
                                      (spike_times < trial_start + baseline_window[1])]
        # Find spikes in response window
        response_spikes = spike_times[(spike_times >= trial_start + response_window[0]) & 
                                      (spike_times < trial_start + response_window[1])]
        
        baseline_duration = baseline_window[1] - baseline_window[0]
        response_duration = response_window[1] - response_window[0]
        
        trial_baseline_rates.append(len(baseline_spikes) / baseline_duration)
        trial_response_rates.append(len(response_spikes) / response_duration)
    
    # Perform paired t-test
    _, p_value = stats.ttest_rel(trial_baseline_rates, trial_response_rates)
    p_values.append(p_value)
    
    # Plot PSTH
    if i < 9:  # Plot first 9 units in a 3x3 grid
        plt.subplot(3, 3, i + 1)
        plt.fill_between(time_centers, mean_psth - sem_psth, mean_psth + sem_psth, alpha=0.3)
        plt.plot(time_centers, mean_psth)
        plt.axvline(x=0, linestyle='--', color='r', label='Stimulus Onset')
        plt.axvline(x=2.0, linestyle='--', color='g', label='Stimulus Offset')
        plt.axhline(y=baseline_rate, linestyle=':', color='k', label='Baseline Rate')
        plt.title(f'Unit {unit_id} (Type {cell_type})')
        plt.xlabel('Time (s)')
        plt.ylabel('Firing Rate (Hz)')
        if i == 0:
            plt.legend(loc='upper right')
        plt.grid(True)

plt.tight_layout()
plt.savefig('tmp_scripts/psth_examples.png')

# Create a figure to compare responses between cell types
cell_type_1 = np.array(cell_types) == 1.0
cell_type_2 = np.array(cell_types) == 2.0

# Calculate average PSTHs for each cell type
if np.sum(cell_type_1) > 0:
    mean_psth_type1 = np.mean([mean_psths[i] for i in range(len(mean_psths)) if cell_type_1[i]], axis=0)
else:
    mean_psth_type1 = np.zeros_like(time_centers)

if np.sum(cell_type_2) > 0:
    mean_psth_type2 = np.mean([mean_psths[i] for i in range(len(mean_psths)) if cell_type_2[i]], axis=0)
else:
    mean_psth_type2 = np.zeros_like(time_centers)

plt.figure(figsize=(10, 6))
plt.plot(time_centers, mean_psth_type1, label='Cell Type 1.0')
plt.plot(time_centers, mean_psth_type2, label='Cell Type 2.0')
plt.axvline(x=0, linestyle='--', color='r', label='Stimulus Onset')
plt.axvline(x=2.0, linestyle='--', color='g', label='Stimulus Offset')
plt.title('Average Response by Cell Type')
plt.xlabel('Time (s)')
plt.ylabel('Firing Rate (Hz)')
plt.legend()
plt.grid(True)
plt.savefig('tmp_scripts/cell_type_comparison.png')

# Create a bar plot comparing baseline vs response rates
plt.figure(figsize=(12, 6))

# Prepare data for plotting
x = np.arange(len(unit_ids))
width = 0.35

plt.bar(x - width/2, baseline_rates, width, label='Baseline Rate')
plt.bar(x + width/2, response_rates, width, label='Response Rate')

# Add significant markers
for i, p in enumerate(p_values):
    if p < 0.05:  # Significant change
        plt.text(i, max(baseline_rates[i], response_rates[i]) + 2, '*', 
                 horizontalalignment='center', fontsize=12)

plt.xlabel('Unit ID')
plt.ylabel('Firing Rate (Hz)')
plt.title('Baseline vs Response Rates')
plt.xticks(x, unit_ids, rotation=90)
plt.legend()
plt.tight_layout()
plt.savefig('tmp_scripts/baseline_vs_response.png')

# Print summary stats
print("\nResponse Statistics:")
print(f"Units with significant response (p<0.05): {np.sum(np.array(p_values) < 0.05)} out of {len(p_values)}")

# Calculate response ratio (response rate / baseline rate)
response_ratio = np.array(response_rates) / np.array(baseline_rates)
print(f"Mean response ratio (response/baseline): {np.mean(response_ratio):.2f}")

# Compare response between cell types
if np.sum(cell_type_1) > 0 and np.sum(cell_type_2) > 0:
    response_ratio_type1 = response_ratio[cell_type_1]
    response_ratio_type2 = response_ratio[cell_type_2]
    
    print(f"\nCell Type 1.0 mean response ratio: {np.mean(response_ratio_type1):.2f}")
    print(f"Cell Type 2.0 mean response ratio: {np.mean(response_ratio_type2):.2f}")
    
    # Statistical comparison between cell types
    _, p_value_types = stats.ttest_ind(response_ratio_type1, response_ratio_type2)
    print(f"p-value for difference between cell types: {p_value_types:.4f}")
else:
    print("Not enough units of both cell types in the sample to compare")

print("Plots saved to tmp_scripts directory")