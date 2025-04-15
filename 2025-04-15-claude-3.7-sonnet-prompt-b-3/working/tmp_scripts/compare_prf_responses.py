"""
This script compares neural responses between two different PRFs (Pulse Repetition Frequencies):
1. 3000 Hz (from first file)
2. 1500 Hz (from second file)

We want to understand:
1. If different PRFs elicit different neural responses
2. If any cell types are more responsive to specific PRFs
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

# URLs for the two files with different PRFs
url_3000hz = "https://api.dandiarchive.org/api/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/download/"
url_1500hz = "https://api.dandiarchive.org/api/assets/526c681d-0c50-44e1-92be-9c0134c71fd8/download/"

print("Loading 3000 Hz PRF data...")
remote_file_3000 = remfile.File(url_3000hz)
h5_file_3000 = h5py.File(remote_file_3000)
io_3000 = pynwb.NWBHDF5IO(file=h5_file_3000)
nwb_3000 = io_3000.read()

print("Loading 1500 Hz PRF data...")
remote_file_1500 = remfile.File(url_1500hz)
h5_file_1500 = h5py.File(remote_file_1500)
io_1500 = pynwb.NWBHDF5IO(file=h5_file_1500)
nwb_1500 = io_1500.read()

# Get basic info about the datasets
print("\nPRF 3000 Hz - Identifier:", nwb_3000.identifier)
print("PRF 1500 Hz - Identifier:", nwb_1500.identifier)

print("\nExtracting trials and units data...")
trials_3000 = nwb_3000.trials.to_dataframe()
trials_1500 = nwb_1500.trials.to_dataframe()
units_3000 = nwb_3000.units.to_dataframe()
units_1500 = nwb_1500.units.to_dataframe()

print(f"3000 Hz: {len(trials_3000)} trials, {len(units_3000)} units")
print(f"1500 Hz: {len(trials_1500)} trials, {len(units_1500)} units")

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

# Function to compute response metrics
def compute_response_metrics(spike_times, trial_times, baseline_window, response_window):
    baseline_rates = []
    response_rates = []
    
    for trial_start in trial_times:
        # Find spikes in baseline window
        baseline_spikes = spike_times[(spike_times >= trial_start + baseline_window[0]) & 
                                      (spike_times < trial_start + baseline_window[1])]
        # Find spikes in response window
        response_spikes = spike_times[(spike_times >= trial_start + response_window[0]) & 
                                      (spike_times < trial_start + response_window[1])]
        
        baseline_duration = baseline_window[1] - baseline_window[0]
        response_duration = response_window[1] - response_window[0]
        
        baseline_rates.append(len(baseline_spikes) / baseline_duration)
        response_rates.append(len(response_spikes) / response_duration)
    
    mean_baseline = np.mean(baseline_rates)
    mean_response = np.mean(response_rates)
    
    # Statistical test
    t_stat, p_value = stats.ttest_rel(baseline_rates, response_rates)
    
    return mean_baseline, mean_response, p_value

# Extract trial start times
trial_starts_3000 = trials_3000['start_time'].values
trial_starts_1500 = trials_1500['start_time'].values

# Set analysis windows
baseline_window = (-0.9, -0.1)  # 0.8s window before stimulus
response_window = (0.1, 0.9)    # 0.8s window during stimulus

# We'll analyze matched units (assuming they're in the same order in both files)
# Sample a subset of units to avoid timeout
n_units_to_analyze = min(10, min(len(units_3000), len(units_1500)))
np.random.seed(42)
unit_indices = np.random.choice(min(len(units_3000), len(units_1500)), size=n_units_to_analyze, replace=False)

# Arrays to store results
unit_ids = []
cell_types = []
baseline_3000 = []
response_3000 = []
p_values_3000 = []
baseline_1500 = []
response_1500 = []
p_values_1500 = []
response_ratio_3000 = []
response_ratio_1500 = []

# Compute average PSTHs for each PRF, separated by cell type
mean_psth_3000_type1 = np.zeros(n_bins)
mean_psth_3000_type2 = np.zeros(n_bins)
mean_psth_1500_type1 = np.zeros(n_bins)
mean_psth_1500_type2 = np.zeros(n_bins)
count_type1 = 0
count_type2 = 0

print("\nAnalyzing matched units across PRFs...")
for unit_idx in unit_indices:
    unit_id_3000 = units_3000.index[unit_idx]
    unit_id_1500 = units_1500.index[unit_idx]
    
    unit_ids.append(unit_id_3000)  # Use ID from first file
    
    spike_times_3000 = units_3000.loc[unit_id_3000, 'spike_times']
    spike_times_1500 = units_1500.loc[unit_id_1500, 'spike_times']
    
    # Get cell type (should be the same across recordings)
    cell_type = units_3000.loc[unit_id_3000, 'celltype_label']
    cell_types.append(cell_type)
    
    # Compute response metrics for 3000 Hz
    bl_3000, resp_3000, p_3000 = compute_response_metrics(
        spike_times_3000, trial_starts_3000, baseline_window, response_window)
    
    # Compute response metrics for 1500 Hz
    bl_1500, resp_1500, p_1500 = compute_response_metrics(
        spike_times_1500, trial_starts_1500, baseline_window, response_window)
    
    # Store results
    baseline_3000.append(bl_3000)
    response_3000.append(resp_3000)
    p_values_3000.append(p_3000)
    
    baseline_1500.append(bl_1500)
    response_1500.append(resp_1500)
    p_values_1500.append(p_1500)
    
    # Compute response ratios
    ratio_3000 = resp_3000 / bl_3000 if bl_3000 > 0 else 0
    ratio_1500 = resp_1500 / bl_1500 if bl_1500 > 0 else 0
    response_ratio_3000.append(ratio_3000)
    response_ratio_1500.append(ratio_1500)
    
    # Compute PSTHs
    mean_psth_3000, _ = compute_psth(spike_times_3000, trial_starts_3000, pre_time, post_time, bin_size)
    mean_psth_1500, _ = compute_psth(spike_times_1500, trial_starts_1500, pre_time, post_time, bin_size)
    
    # Add to the average PSTHs by cell type
    if cell_type == 1.0:
        mean_psth_3000_type1 += mean_psth_3000
        mean_psth_1500_type1 += mean_psth_1500
        count_type1 += 1
    elif cell_type == 2.0:
        mean_psth_3000_type2 += mean_psth_3000
        mean_psth_1500_type2 += mean_psth_1500
        count_type2 += 1

# Compute averages
if count_type1 > 0:
    mean_psth_3000_type1 /= count_type1
    mean_psth_1500_type1 /= count_type1

if count_type2 > 0:
    mean_psth_3000_type2 /= count_type2
    mean_psth_1500_type2 /= count_type2

# Plot average PSTHs by cell type and PRF
plt.figure(figsize=(12, 8))

# Cell Type 1
plt.subplot(2, 1, 1)
plt.plot(time_centers, mean_psth_3000_type1, 'b-', label='3000 Hz')
plt.plot(time_centers, mean_psth_1500_type1, 'r-', label='1500 Hz')
plt.axvline(x=0, linestyle='--', color='k', label='Stimulus Onset')
plt.axvline(x=2.0, linestyle='--', color='g', label='Stimulus Offset')
plt.title('Cell Type 1.0 - Average Response by PRF')
plt.xlabel('Time (s)')
plt.ylabel('Firing Rate (Hz)')
plt.legend()
plt.grid(True)

# Cell Type 2
plt.subplot(2, 1, 2)
plt.plot(time_centers, mean_psth_3000_type2, 'b-', label='3000 Hz')
plt.plot(time_centers, mean_psth_1500_type2, 'r-', label='1500 Hz')
plt.axvline(x=0, linestyle='--', color='k')
plt.axvline(x=2.0, linestyle='--', color='g')
plt.title('Cell Type 2.0 - Average Response by PRF')
plt.xlabel('Time (s)')
plt.ylabel('Firing Rate (Hz)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('tmp_scripts/prf_comparison_by_cell_type.png')

# Plot response ratios to compare PRFs
plt.figure(figsize=(10, 6))
x = np.arange(len(unit_ids))
width = 0.35

plt.bar(x - width/2, response_ratio_3000, width, label='3000 Hz PRF')
plt.bar(x + width/2, response_ratio_1500, width, label='1500 Hz PRF')

# Add significance markers
for i, (p3000, p1500) in enumerate(zip(p_values_3000, p_values_1500)):
    if p3000 < 0.05:  # Significant response for 3000 Hz
        plt.text(i - width/2, response_ratio_3000[i] + 0.05, '*', 
                 horizontalalignment='center', fontsize=12)
    if p1500 < 0.05:  # Significant response for 1500 Hz
        plt.text(i + width/2, response_ratio_1500[i] + 0.05, '*', 
                 horizontalalignment='center', fontsize=12)

plt.xlabel('Unit ID')
plt.ylabel('Response Ratio (Response/Baseline)')
plt.title('Response Modulation by PRF')
plt.axhline(y=1.0, linestyle='--', color='k', label='No Change')
plt.xticks(x, unit_ids, rotation=90)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('tmp_scripts/response_ratio_by_prf.png')

# Plot scatter comparison of response ratios
plt.figure(figsize=(8, 8))
plt.scatter(response_ratio_3000, response_ratio_1500, alpha=0.7)
plt.plot([0.8, 1.2], [0.8, 1.2], 'k--')  # Plot y=x line

for i, unit_id in enumerate(unit_ids):
    plt.annotate(f"{unit_id}", (response_ratio_3000[i], response_ratio_1500[i]),
                fontsize=9, alpha=0.7)

plt.xlim(0.8, 1.2)
plt.ylim(0.8, 1.2)
plt.axhline(y=1, color='k', linestyle=':', alpha=0.5)
plt.axvline(x=1, color='k', linestyle=':', alpha=0.5)
plt.title('Comparing Unit Responses Between PRFs')
plt.xlabel('Response Ratio (3000 Hz)')
plt.ylabel('Response Ratio (1500 Hz)')
plt.grid(True)
plt.savefig('tmp_scripts/response_correlation.png')

# Statistical comparison
print("\nPRF Response Comparison Statistics:")
print(f"Mean response ratio for 3000 Hz: {np.mean(response_ratio_3000):.3f}")
print(f"Mean response ratio for 1500 Hz: {np.mean(response_ratio_1500):.3f}")

# Statistical test comparing response ratios between PRFs
t_stat, p_value = stats.ttest_rel(response_ratio_3000, response_ratio_1500)
print(f"Paired t-test p-value: {p_value:.4f}")

if p_value < 0.05:
    print("There is a statistically significant difference in neural responses between the two PRFs")
else:
    print("No statistically significant difference detected in neural responses between PRFs")

# Compare responses by cell type
cell_type_1_mask = np.array(cell_types) == 1.0
cell_type_2_mask = np.array(cell_types) == 2.0

if np.sum(cell_type_1_mask) > 0:
    print(f"\nCell Type 1.0 (n={np.sum(cell_type_1_mask)}):")
    print(f"  Mean response ratio for 3000 Hz: {np.mean(np.array(response_ratio_3000)[cell_type_1_mask]):.3f}")
    print(f"  Mean response ratio for 1500 Hz: {np.mean(np.array(response_ratio_1500)[cell_type_1_mask]):.3f}")

if np.sum(cell_type_2_mask) > 0:
    print(f"\nCell Type 2.0 (n={np.sum(cell_type_2_mask)}):")
    print(f"  Mean response ratio for 3000 Hz: {np.mean(np.array(response_ratio_3000)[cell_type_2_mask]):.3f}")
    print(f"  Mean response ratio for 1500 Hz: {np.mean(np.array(response_ratio_1500)[cell_type_2_mask]):.3f}")

print("\nAnalysis completed and plots saved in tmp_scripts directory.")