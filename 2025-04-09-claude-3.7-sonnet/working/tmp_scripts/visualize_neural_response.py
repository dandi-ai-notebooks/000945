# This script visualizes neural responses to ultrasound stimulation by creating
# raster plots and PSTHs (Peri-Stimulus Time Histograms) for selected units

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set plotting style
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'axes.grid': True})

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/655fe6cf-a152-412b-9d20-71c6db670629/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

print(f"Loading data from NWB file: {nwb.identifier}")
print(f"Session description: {nwb.session_description}")
print(f"Subject: {nwb.subject.subject_id}")

# Get trial timing information
n_trials = len(nwb.trials.id[:])
trial_starts = nwb.trials['start_time'][:]
trial_stops = nwb.trials['stop_time'][:]
trial_durations = trial_stops - trial_starts

print(f"Number of trials: {n_trials}")
print(f"Average trial duration: {np.mean(trial_durations):.3f} seconds")

# Define time window around trial onset for analysis (in seconds)
pre_time = 1.0  # time before trial onset
post_time = 2.0  # time after trial onset

# Select a subset of units to visualize (first 5 units)
unit_ids = list(range(5))

# Create a figure for raster plot and PSTH
fig = plt.figure(figsize=(12, 10))
gs = GridSpec(len(unit_ids), 2, width_ratios=[3, 1], figure=fig)
fig.suptitle(f"Neural responses to ultrasound stimulation\n{nwb.identifier}", fontsize=14)

# Create bins for PSTH
bin_size = 0.05  # 50 ms
bins = np.arange(-pre_time, post_time + bin_size, bin_size)
bin_centers = bins[:-1] + bin_size/2

# Process each selected unit
for i, unit_id in enumerate(unit_ids):
    print(f"Processing unit {unit_id}...")
    
    # Get spike times for this unit
    spike_times = nwb.units['spike_times'][unit_id][:]
    
    # Get cell type (if available)
    cell_type = "Unknown"
    if 'celltype_label' in nwb.units.colnames:
        cell_type_value = nwb.units['celltype_label'][unit_id]
        cell_type = f"Type {cell_type_value}"
    
    # Create trial-aligned raster plot
    raster_ax = fig.add_subplot(gs[i, 0])
    
    # Store spike counts for PSTH
    all_trial_counts = []
    
    # Loop through a subset of trials (first 50 for clarity in the plot)
    max_trials_to_plot = min(50, n_trials)
    
    for trial_idx in range(max_trials_to_plot):
        # Get trial onset time
        trial_onset = trial_starts[trial_idx]
        
        # Find spikes within the time window relative to this trial
        trial_mask = (spike_times >= trial_onset - pre_time) & (spike_times <= trial_onset + post_time)
        trial_spikes = spike_times[trial_mask] - trial_onset  # align to trial onset
        
        # Plot raster for this trial
        raster_ax.plot(trial_spikes, np.ones_like(trial_spikes) * (trial_idx + 1), '|', color='black', markersize=4)
        
        # Compute histogram for this trial for the PSTH
        counts, _ = np.histogram(trial_spikes, bins=bins)
        all_trial_counts.append(counts)
    
    # Format raster plot
    raster_ax.set_ylabel(f"Trial #")
    if i == len(unit_ids) - 1:
        raster_ax.set_xlabel("Time from trial onset (s)")
    
    raster_ax.set_xlim(-pre_time, post_time)
    raster_ax.set_ylim(0, max_trials_to_plot + 1)
    raster_ax.axvline(x=0, color='r', linestyle='--', label='Stim onset')
    
    if i == 0:
        raster_ax.set_title(f"Spike raster (first {max_trials_to_plot} trials)")
    
    # Add unit label
    raster_ax.text(0.02, 0.95, f"Unit {unit_id} ({cell_type})", transform=raster_ax.transAxes,
                  verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Create PSTH (peri-stimulus time histogram)
    psth_ax = fig.add_subplot(gs[i, 1])
    
    # Calculate mean firing rate across trials
    mean_counts = np.mean(all_trial_counts, axis=0)
    mean_rate = mean_counts / bin_size  # Convert to Hz
    
    # Plot PSTH
    psth_ax.bar(bin_centers, mean_rate, width=bin_size * 0.9, alpha=0.7)
    psth_ax.axvline(x=0, color='r', linestyle='--')
    
    psth_ax.set_ylabel("Rate (Hz)")
    if i == len(unit_ids) - 1:
        psth_ax.set_xlabel("Time (s)")
    
    if i == 0:
        psth_ax.set_title("PSTH")
    
    # Make y-axis limits reasonable
    y_max = np.ceil(np.max(mean_rate) * 1.1)
    psth_ax.set_ylim(0, y_max)
    psth_ax.set_xlim(-pre_time, post_time)

plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.3)

# Save the figure
plt.savefig("tmp_scripts/neural_response_raster_psth.png", dpi=150, bbox_inches="tight")
print("Figure saved to tmp_scripts/neural_response_raster_psth.png")

# Create a figure showing overall population response
plt.figure(figsize=(10, 6))

# Get all units and their cell types
n_units = len(nwb.units.id[:])
cell_types = np.array([nwb.units['celltype_label'][i] for i in range(n_units)])

# Group units by cell type
type1_units = np.where(cell_types == 1.0)[0]
type2_units = np.where(cell_types == 2.0)[0]

print(f"\nFound {len(type1_units)} units of Type 1.0")
print(f"Found {len(type2_units)} units of Type 2.0")

# Analyze population response for each cell type
for cell_type, unit_group, color, label in [
    (1.0, type1_units[:20], 'blue', 'Type 1.0'),  # Limit to first 20 units for each type
    (2.0, type2_units[:20], 'red', 'Type 2.0')
]:
    # Store all PSTHs
    all_psths = []
    
    # Process each unit in this group
    for unit_id in unit_group:
        # Get spike times for this unit
        spike_times = nwb.units['spike_times'][unit_id][:]
        
        # Initialize array to store trial-aligned spike counts
        trial_counts = np.zeros((n_trials, len(bins)-1))
        
        # Process each trial
        for trial_idx in range(n_trials):
            # Get trial onset time
            trial_onset = trial_starts[trial_idx]
            
            # Find spikes within time window relative to this trial
            trial_mask = (spike_times >= trial_onset - pre_time) & (spike_times <= trial_onset + post_time)
            trial_spikes = spike_times[trial_mask] - trial_onset  # align to trial onset
            
            # Compute histogram for this trial
            counts, _ = np.histogram(trial_spikes, bins=bins)
            trial_counts[trial_idx] = counts
        
        # Calculate mean firing rate across trials for this unit
        mean_counts = np.mean(trial_counts, axis=0)
        mean_rate = mean_counts / bin_size  # Convert to Hz
        
        # Z-score normalize to compare units with different baseline firing rates
        baseline = mean_rate[bin_centers < 0]  # Use pre-stimulus period as baseline
        baseline_mean = np.mean(baseline)
        baseline_std = np.std(baseline) if np.std(baseline) > 0 else 1.0  # Avoid division by zero
        
        normalized_rate = (mean_rate - baseline_mean) / baseline_std
        all_psths.append(normalized_rate)
    
    # Calculate average response across units of this type
    if all_psths:
        avg_response = np.mean(all_psths, axis=0)
        sem_response = np.std(all_psths, axis=0) / np.sqrt(len(all_psths))  # Standard error of mean
        
        # Plot average response with shaded error
        plt.plot(bin_centers, avg_response, color=color, label=f"{label} (n={len(unit_group)})")
        plt.fill_between(bin_centers, avg_response - sem_response, avg_response + sem_response, 
                         color=color, alpha=0.2)

# Add plot details
plt.axvline(x=0, color='black', linestyle='--', label='Stim onset')
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
plt.title(f"Population response to ultrasound stimulation\n{nwb.identifier}")
plt.xlabel("Time from stimulation onset (s)")
plt.ylabel("Normalized firing rate (z-score)")
plt.legend()
plt.grid(True)
plt.xlim(-pre_time, post_time)
plt.tight_layout()

# Save the figure
plt.savefig("tmp_scripts/population_response.png", dpi=150, bbox_inches="tight")
print("Figure saved to tmp_scripts/population_response.png")

# Close files
io.close()
f.close()
file.close()