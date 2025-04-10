# This script compares neural responses to different ultrasound stimulation frequencies (PRFs)
# for the same subject (BH506). We compare 1500 Hz and 4500 Hz PRFs.

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Set plotting style
plt.rcParams.update({'font.size': 12})
plt.rcParams.update({'axes.grid': True})

# URLs and identifiers for the two files with different PRFs
file_info = [
    {
        "url": "https://api.dandiarchive.org/api/assets/655fe6cf-a152-412b-9d20-71c6db670629/download/",
        "id": "BH506_1500_200",
        "prf": "1500 Hz",
        "color": "blue"
    },
    {
        "url": "https://api.dandiarchive.org/api/assets/b353fb55-5a3d-4961-81a2-c121f31c5344/download/",
        "id": "BH506_4500_200",
        "prf": "4500 Hz",
        "color": "red"
    }
]

# Define time window around trial onset for analysis (in seconds)
pre_time = 1.0  # time before trial onset
post_time = 2.0  # time after trial onset

# Create bins for PSTH
bin_size = 0.05  # 50 ms
bins = np.arange(-pre_time, post_time + bin_size, bin_size)
bin_centers = bins[:-1] + bin_size/2

# Create figure for comparing population responses
plt.figure(figsize=(12, 10))

# Process each file (different PRF condition)
for file_data in file_info:
    print(f"\nProcessing file: {file_data['id']} (PRF: {file_data['prf']})")
    
    # Load the NWB file
    file = remfile.File(file_data['url'])
    f = h5py.File(file)
    io = pynwb.NWBHDF5IO(file=f)
    nwb = io.read()
    
    # Get basic info
    print(f"Session description: {nwb.session_description}")
    print(f"Subject: {nwb.subject.subject_id}")
    
    # Get trial timing information
    n_trials = len(nwb.trials.id[:])
    trial_starts = nwb.trials['start_time'][:]
    trial_stops = nwb.trials['stop_time'][:]
    trial_durations = trial_stops - trial_starts
    
    print(f"Number of trials: {n_trials}")
    print(f"Average trial duration: {np.mean(trial_durations):.3f} seconds")
    
    # Get all units and their cell types
    n_units = len(nwb.units.id[:])
    cell_types = np.array([nwb.units['celltype_label'][i] for i in range(n_units)])
    
    # Count the number of each cell type
    type1_units = np.where(cell_types == 1.0)[0]
    type2_units = np.where(cell_types == 2.0)[0]
    
    print(f"Total units: {n_units}")
    print(f"Type 1.0 units: {len(type1_units)}")
    print(f"Type 2.0 units: {len(type2_units)}")
    
    # Create subplot to compare overall population response between conditions
    plt.subplot(2, 1, file_info.index(file_data) + 1)
    
    # Process each cell type
    for cell_type_value, label, line_style in [
        (1.0, "Type 1.0", "-"), 
        (2.0, "Type 2.0", "--")
    ]:
        # Get units of this type
        units_of_type = np.where(cell_types == cell_type_value)[0]
        
        # Limit the units to prevent memory issues (max 20)
        units_to_analyze = units_of_type[:min(20, len(units_of_type))]
        
        # Skip if no units of this type
        if len(units_to_analyze) == 0:
            continue
            
        print(f"Analyzing {len(units_to_analyze)} units of {label}")
        
        # Store all PSTHs
        all_psths = []
        
        # Process each unit
        for unit_id in units_to_analyze:
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
            plt.plot(bin_centers, avg_response, color=file_data["color"], 
                     linestyle=line_style, 
                     label=f"{file_data['prf']} - {label} (n={len(units_to_analyze)})")
            plt.fill_between(bin_centers, avg_response - sem_response, avg_response + sem_response, 
                             color=file_data["color"], alpha=0.2)
    
    # Add plot details
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.7, label='Stim onset')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.title(f"Neural response to {file_data['prf']} ultrasound stimulation")
    plt.xlabel("Time from stimulation onset (s)")
    plt.ylabel("Normalized firing rate (z-score)")
    plt.legend(loc='upper right')
    plt.xlim(-pre_time, post_time)
    plt.tight_layout()
    
    # Close files
    io.close()
    f.close()
    file.close()

# Adjust layout
plt.suptitle("Comparison of Neural Responses to Different Ultrasound Frequencies (PRFs)", fontsize=16)
plt.subplots_adjust(top=0.93, hspace=0.3)

# Save the figure
plt.savefig("tmp_scripts/prf_comparison.png", dpi=150, bbox_inches="tight")
print("\nFigure saved to tmp_scripts/prf_comparison.png")

# Create a combined figure to directly compare the two PRFs
plt.figure(figsize=(12, 8))

# Reload and reprocess the data for the combined figure
cell_type_results = {}

for file_data in file_info:
    print(f"\nReprocessing file: {file_data['id']} for combined figure")
    
    # Load the NWB file
    file = remfile.File(file_data['url'])
    f = h5py.File(file)
    io = pynwb.NWBHDF5IO(file=f)
    nwb = io.read()
    
    # Get trial timing information
    trial_starts = nwb.trials['start_time'][:]
    n_trials = len(trial_starts)
    
    # Get all units and their cell types
    n_units = len(nwb.units.id[:])
    cell_types = np.array([nwb.units['celltype_label'][i] for i in range(n_units)])
    
    # Process each cell type
    for cell_type_value in [1.0, 2.0]:
        key = f"Type {cell_type_value}"
        if key not in cell_type_results:
            cell_type_results[key] = {}
            
        # Get units of this type
        units_of_type = np.where(cell_types == cell_type_value)[0]
        
        # Limit the units to prevent memory issues (max 20)
        units_to_analyze = units_of_type[:min(20, len(units_of_type))]

        # Skip if no units of this type
        if len(units_to_analyze) == 0:
            continue
            
        # Store all PSTHs
        all_psths = []
        
        # Process each unit
        for unit_id in units_to_analyze:
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
            
            # Store the results
            cell_type_results[key][file_data['prf']] = {
                'avg': avg_response,
                'sem': sem_response,
                'n': len(units_to_analyze),
                'color': file_data['color']
            }
    
    # Close files
    io.close()
    f.close()
    file.close()

# Plot combined figure
for i, cell_type in enumerate(cell_type_results.keys()):
    plt.subplot(1, 2, i+1)
    
    for prf in ['1500 Hz', '4500 Hz']:
        if prf in cell_type_results[cell_type]:
            data = cell_type_results[cell_type][prf]
            plt.plot(bin_centers, data['avg'], color=data['color'], 
                     label=f"{prf} (n={data['n']})")
            plt.fill_between(bin_centers, data['avg'] - data['sem'], data['avg'] + data['sem'], 
                            color=data['color'], alpha=0.2)
    
    # Add plot details
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.7, label='Stim onset')
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    plt.title(f"{cell_type} Response Comparison")
    plt.xlabel("Time from stimulation onset (s)")
    plt.ylabel("Normalized firing rate (z-score)")
    plt.legend(loc='upper right')
    plt.xlim(-pre_time, post_time)
    plt.grid(True)

plt.suptitle("Direct Comparison of Neural Responses by PRF and Cell Type", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# Save the figure
plt.savefig("tmp_scripts/prf_direct_comparison.png", dpi=150, bbox_inches="tight")
print("Figure saved to tmp_scripts/prf_direct_comparison.png")