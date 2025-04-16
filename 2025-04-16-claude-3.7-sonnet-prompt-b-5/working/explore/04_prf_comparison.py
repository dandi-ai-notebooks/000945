"""
This script compares neural responses to two different pulse repetition frequencies (PRFs)
of transcranial focused ultrasound stimulation: 30 Hz and 1500 Hz.
"""

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Define URLs for the two different PRF files
url_1500hz = "https://api.dandiarchive.org/api/assets/b88188c8-4e4c-494c-8dab-806b1efd55eb/download/"
url_30hz = "https://api.dandiarchive.org/api/assets/ce7bcbee-3dfa-4672-b8a2-b60aff02e32e/download/"

# Function to load NWB file
def load_nwb(url):
    remote_file = remfile.File(url)
    h5_file = h5py.File(remote_file)
    io = pynwb.NWBHDF5IO(file=h5_file)
    nwb = io.read()
    return nwb, h5_file, remote_file, io

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

# Function to analyze a dataset with a specific PRF
def analyze_prf_dataset(url, prf_label):
    print(f"\n=== Analyzing {prf_label} PRF dataset ===")
    
    # Load NWB file
    nwb, h5_file, remote_file, io = load_nwb(url)
    
    # Print basic information
    print(f"Identifier: {nwb.identifier}")
    print(f"Session Description: {nwb.session_description}")
    print(f"Session Start Time: {nwb.session_start_time}")
    
    # Get trials dataframe
    trials_df = nwb.trials.to_dataframe()
    trial_starts = trials_df['start_time'].values
    trial_stops = trials_df['stop_time'].values
    
    print(f"Number of trials: {len(trials_df)}")
    print(f"Trial duration: {np.mean(trial_stops - trial_starts):.6f} seconds")
    
    # Get units dataframe
    units_df = nwb.units.to_dataframe()
    
    print(f"Number of units: {len(units_df)}")
    if 'celltype_label' in units_df.columns:
        rsu_count = len(units_df[units_df['celltype_label'] == 1])
        fsu_count = len(units_df[units_df['celltype_label'] == 2])
        print(f"RSU count: {rsu_count}")
        print(f"FSU count: {fsu_count}")
    
    # Define time windows for analysis
    pre_window = 1.0  # 1 second before stimulus
    post_window = 1.0  # 1 second after stimulus
    bin_size = 0.05  # 50 ms bins
    
    # Split units by cell type
    rsu_units = units_df[units_df['celltype_label'] == 1]
    fsu_units = units_df[units_df['celltype_label'] == 2]
    
    # Compute average responses for each cell type
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

    # Close resources
    io.close()
    h5_file.close()
    remote_file.close()
    
    # Calculate average responses
    rsu_mean = np.mean(rsu_avg_rates, axis=0) if rsu_avg_rates else np.zeros_like(bin_centers)
    fsu_mean = np.mean(fsu_avg_rates, axis=0) if fsu_avg_rates else np.zeros_like(bin_centers)
    
    # Return results
    return {
        'bin_centers': bin_centers,
        'rsu_mean': rsu_mean,
        'fsu_mean': fsu_mean,
        'trial_duration': np.mean(trial_stops - trial_starts)
    }

# Analyze both datasets
results_1500hz = analyze_prf_dataset(url_1500hz, "1500 Hz")
results_30hz = analyze_prf_dataset(url_30hz, "30 Hz")

# Compare RSU responses between PRFs
plt.figure(figsize=(12, 5))

# First subplot: RSU comparison
plt.subplot(1, 2, 1)
plt.plot(results_1500hz['bin_centers'], results_1500hz['rsu_mean'], 'b-', label='1500 Hz PRF', linewidth=2)
plt.plot(results_30hz['bin_centers'], results_30hz['rsu_mean'], 'g-', label='30 Hz PRF', linewidth=2)
plt.axvline(x=0, color='k', linestyle='--', label='Stimulus Onset')
# Add stimulus offset for each PRF
plt.axvline(x=results_1500hz['trial_duration'], color='b', linestyle='--', label='1500 Hz Stimulus Offset')
plt.axvline(x=results_30hz['trial_duration'], color='g', linestyle='--', label='30 Hz Stimulus Offset')
plt.xlabel('Time relative to trial start (s)')
plt.ylabel('Average Firing Rate (spikes/s)')
plt.title('RSU Responses: 30 Hz vs 1500 Hz PRF')
plt.legend()
plt.grid(True, alpha=0.3)

# Second subplot: FSU comparison
plt.subplot(1, 2, 2)
plt.plot(results_1500hz['bin_centers'], results_1500hz['fsu_mean'], 'b-', label='1500 Hz PRF', linewidth=2)
plt.plot(results_30hz['bin_centers'], results_30hz['fsu_mean'], 'g-', label='30 Hz PRF', linewidth=2)
plt.axvline(x=0, color='k', linestyle='--', label='Stimulus Onset')
# Add stimulus offset for each PRF
plt.axvline(x=results_1500hz['trial_duration'], color='b', linestyle='--', label='1500 Hz Stimulus Offset')
plt.axvline(x=results_30hz['trial_duration'], color='g', linestyle='--', label='30 Hz Stimulus Offset')
plt.xlabel('Time relative to trial start (s)')
plt.ylabel('Average Firing Rate (spikes/s)')
plt.title('FSU Responses: 30 Hz vs 1500 Hz PRF')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('explore/prf_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Calculate response magnitude (change in firing rate from baseline)
def calculate_response_magnitude(results):
    # Define baseline period (average of first 0.5 seconds before stimulation)
    baseline_indices = (results['bin_centers'] >= -1.0) & (results['bin_centers'] < -0.5)
    stim_indices = (results['bin_centers'] >= 0.0) & (results['bin_centers'] < 0.5)
    
    # Calculate baseline and stimulation period averages
    rsu_baseline = np.mean(results['rsu_mean'][baseline_indices])
    rsu_stim = np.mean(results['rsu_mean'][stim_indices])
    fsu_baseline = np.mean(results['fsu_mean'][baseline_indices])
    fsu_stim = np.mean(results['fsu_mean'][stim_indices])
    
    # Calculate percent change from baseline
    rsu_percent_change = ((rsu_stim - rsu_baseline) / rsu_baseline) * 100 if rsu_baseline != 0 else 0
    fsu_percent_change = ((fsu_stim - fsu_baseline) / fsu_baseline) * 100 if fsu_baseline != 0 else 0
    
    return {
        'rsu_baseline': rsu_baseline,
        'rsu_stim': rsu_stim,
        'rsu_percent_change': rsu_percent_change,
        'fsu_baseline': fsu_baseline,
        'fsu_stim': fsu_stim,
        'fsu_percent_change': fsu_percent_change
    }

# Calculate response magnitudes
mag_1500hz = calculate_response_magnitude(results_1500hz)
mag_30hz = calculate_response_magnitude(results_30hz)

# Plot response magnitude comparison
plt.figure(figsize=(10, 6))
labels = ['RSU - 30 Hz', 'RSU - 1500 Hz', 'FSU - 30 Hz', 'FSU - 1500 Hz']
percents = [mag_30hz['rsu_percent_change'], mag_1500hz['rsu_percent_change'], 
            mag_30hz['fsu_percent_change'], mag_1500hz['fsu_percent_change']]

bar_colors = ['lightgreen', 'lightblue', 'darkgreen', 'darkblue']
plt.bar(labels, percents, color=bar_colors)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.ylabel('Percent Change in Firing Rate from Baseline')
plt.title('Neural Response Magnitude by Cell Type and PRF')
plt.grid(True, alpha=0.3, axis='y')
plt.savefig('explore/response_magnitude_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Print response magnitude results
print("\n=== Response Magnitude Analysis ===")
print(f"1500 Hz PRF:")
print(f"  RSU: {mag_1500hz['rsu_percent_change']:.2f}% change from baseline")
print(f"  FSU: {mag_1500hz['fsu_percent_change']:.2f}% change from baseline")
print(f"30 Hz PRF:")
print(f"  RSU: {mag_30hz['rsu_percent_change']:.2f}% change from baseline")
print(f"  FSU: {mag_30hz['fsu_percent_change']:.2f}% change from baseline")

print("\nPlots saved to explore directory.")