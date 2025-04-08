'''
This script explores the spike data within an NWB file from the Dandiset.
It will:
1. Load a sample NWB file
2. Examine spike times relative to ultrasound stimulation trials
3. Create raster plots and peri-stimulus time histograms
4. Compare responses across different neuron types
'''

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from dandi.dandiapi import DandiAPIClient

# Set up plotting to save to file instead of displaying
plt.ioff()  # Turn interactive mode off

# Choose a sample NWB file to examine
print("Getting a sample NWB file...")
client = DandiAPIClient()
dandiset = client.get_dandiset("000945")
assets = list(dandiset.get_assets())

# Let's pick a file with a reasonable size and from BH498 which has many files
sample_asset = None
for asset in assets:
    if "BH498" in asset.path and 2000000 < asset.size < 10000000:
        sample_asset = asset
        break

if sample_asset is None:
    sample_asset = assets[0]  # Fallback to the first asset if no suitable one found

sample_path = sample_asset.path
sample_id = sample_asset.identifier
print(f"Examining file: {sample_path}")

# Create the Lindi file URL for accessing the NWB file
lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/000945/assets/{sample_id}/nwb.lindi.json"

# Open the NWB file
print("Opening NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print(f"Session description: {nwb.session_description}")
print(f"NWB identifier: {nwb.identifier}")

# Get trials information
trials = nwb.intervals["trials"]
trial_start_times = trials["start_time"][:]
trial_stop_times = trials["stop_time"][:]
print(f"Number of trials: {len(trial_start_times)}")
print(f"Trial duration: {np.mean(trial_stop_times - trial_start_times):.3f} seconds")

# Get units (neurons) information
units = nwb.units
num_units = len(units)
print(f"Number of units: {num_units}")

# Get cell type labels
cell_types = units["celltype_label"].data[:]
num_rsu = np.sum(cell_types == 1)
num_fsu = np.sum(cell_types == 2)
print(f"Number of RSU cells (type 1): {num_rsu}")
print(f"Number of FSU cells (type 2): {num_fsu}")

# Function to create a raster plot for a given unit around trial onsets
def create_raster_plot(unit_index, window=(-0.5, 1.0), num_trials=50):
    spike_times = units["spike_times"][unit_index]
    cell_type = "RSU" if cell_types[unit_index] == 1 else "FSU"
    
    # Limit to first num_trials trials for clarity
    trial_starts = trial_start_times[:num_trials]
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # For each trial, plot the spikes that occur within the window around trial onset
    for i, trial_start in enumerate(trial_starts):
        # Find spikes within the window of this trial
        trial_window_start = trial_start + window[0]
        trial_window_end = trial_start + window[1]
        trial_spikes = spike_times[(spike_times >= trial_window_start) & 
                                   (spike_times <= trial_window_end)]
        
        # Plot each spike as a vertical line at the corresponding time, shifted by trial_start
        if len(trial_spikes) > 0:
            ax.vlines(trial_spikes - trial_start, i + 0.5, i + 1.5, color='black', linewidth=0.5)
    
    # Add a vertical line at t=0 (trial onset)
    ax.axvline(x=0, color='red', linestyle='--', label='Trial Onset')
    
    # Add a vertical line at the mean trial duration
    mean_duration = np.mean(trial_stop_times - trial_start_times)
    ax.axvline(x=mean_duration, color='blue', linestyle='--', label='Trial Offset (Mean)')
    
    # Set labels and title
    ax.set_xlabel('Time relative to trial onset (s)')
    ax.set_ylabel('Trial Number')
    ax.set_title(f'Raster Plot for Unit {unit_index} ({cell_type})')
    ax.set_xlim(window)
    ax.set_ylim(0.5, len(trial_starts) + 0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"tmp_scripts/raster_unit_{unit_index}.png")
    plt.close()

# Function to create a PSTH (Peri-Stimulus Time Histogram) for a given unit
def create_psth(unit_index, window=(-0.5, 1.0), bin_width=0.05):
    spike_times = units["spike_times"][unit_index]
    cell_type = "RSU" if cell_types[unit_index] == 1 else "FSU"
    
    # Create bins for the histogram
    bins = np.arange(window[0], window[1] + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width/2
    
    # Create histogram data
    hist_data = np.zeros(len(bin_centers))
    
    # For each trial, count spikes in each bin
    for trial_start in trial_start_times:
        # Find spikes within the window of this trial
        trial_window_start = trial_start + window[0]
        trial_window_end = trial_start + window[1]
        trial_spikes = spike_times[(spike_times >= trial_window_start) & 
                                   (spike_times <= trial_window_end)]
        
        # Convert spike times to times relative to trial start
        relative_spike_times = trial_spikes - trial_start
        
        # Count spikes in each bin
        hist, _ = np.histogram(relative_spike_times, bins=bins)
        hist_data += hist
    
    # Convert to firing rate (spikes/s)
    firing_rate = hist_data / (len(trial_start_times) * bin_width)
    
    # Create a figure
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    
    # Plot the histogram
    ax.bar(bin_centers, firing_rate, width=bin_width, alpha=0.7, color='blue')
    
    # Add a vertical line at t=0 (trial onset)
    ax.axvline(x=0, color='red', linestyle='--', label='Trial Onset')
    
    # Add a vertical line at the mean trial duration
    mean_duration = np.mean(trial_stop_times - trial_start_times)
    ax.axvline(x=mean_duration, color='green', linestyle='--', label='Trial Offset (Mean)')
    
    # Set labels and title
    ax.set_xlabel('Time relative to trial onset (s)')
    ax.set_ylabel('Firing Rate (spikes/s)')
    ax.set_title(f'PSTH for Unit {unit_index} ({cell_type})')
    ax.set_xlim(window)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"tmp_scripts/psth_unit_{unit_index}.png")
    plt.close()

# Create a figure showing average firing rates before, during, and after stimulation
def plot_average_firing_rates():
    # Define time windows relative to trial onset (in seconds)
    pre_stim = (-0.5, 0)  # 0.5s before stimulation
    during_stim = (0, 0.067)  # During stimulation (67ms)
    post_stim = (0.067, 0.5)  # 0.5s after stimulation
    
    # Arrays to store firing rates
    rsu_rates = np.zeros((num_rsu, 3))  # [pre, during, post]
    fsu_rates = np.zeros((num_fsu, 3))  # [pre, during, post]
    
    # Calculate firing rates for each unit
    rsu_index = 0
    fsu_index = 0
    
    for unit_index in range(num_units):
        spike_times = units["spike_times"][unit_index]
        cell_type = cell_types[unit_index]
        
        # Calculate firing rates for each time window
        pre_rate = 0
        during_rate = 0
        post_rate = 0
        
        for trial_start in trial_start_times:
            # Pre-stimulation window
            window_start = trial_start + pre_stim[0]
            window_end = trial_start + pre_stim[1]
            spikes_in_window = np.sum((spike_times >= window_start) & (spike_times < window_end))
            pre_rate += spikes_in_window / (pre_stim[1] - pre_stim[0])
            
            # During stimulation window
            window_start = trial_start + during_stim[0]
            window_end = trial_start + during_stim[1]
            spikes_in_window = np.sum((spike_times >= window_start) & (spike_times < window_end))
            during_rate += spikes_in_window / (during_stim[1] - during_stim[0])
            
            # Post-stimulation window
            window_start = trial_start + post_stim[0]
            window_end = trial_start + post_stim[1]
            spikes_in_window = np.sum((spike_times >= window_start) & (spike_times < window_end))
            post_rate += spikes_in_window / (post_stim[1] - post_stim[0])
        
        # Average over all trials
        n_trials = len(trial_start_times)
        pre_rate /= n_trials
        during_rate /= n_trials
        post_rate /= n_trials
        
        # Store in appropriate array
        if cell_type == 1:  # RSU
            rsu_rates[rsu_index] = [pre_rate, during_rate, post_rate]
            rsu_index += 1
        elif cell_type == 2:  # FSU
            fsu_rates[fsu_index] = [pre_rate, during_rate, post_rate]
            fsu_index += 1
    
    # Calculate averages across units
    avg_rsu_rates = np.mean(rsu_rates, axis=0)
    avg_fsu_rates = np.mean(fsu_rates, axis=0)
    
    # Calculate standard error of the mean
    sem_rsu_rates = np.std(rsu_rates, axis=0) / np.sqrt(num_rsu)
    sem_fsu_rates = np.std(fsu_rates, axis=0) / np.sqrt(num_fsu)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    x = np.arange(3)
    width = 0.35
    
    plt.bar(x - width/2, avg_rsu_rates, width, yerr=sem_rsu_rates, label='RSU', color='blue', alpha=0.7)
    plt.bar(x + width/2, avg_fsu_rates, width, yerr=sem_fsu_rates, label='FSU', color='orange', alpha=0.7)
    
    plt.xlabel('Time Window')
    plt.ylabel('Average Firing Rate (spikes/s)')
    plt.title('Average Firing Rates By Cell Type')
    plt.xticks(x, ['Pre-Stim', 'During-Stim', 'Post-Stim'])
    plt.legend()
    plt.tight_layout()
    plt.savefig("tmp_scripts/average_firing_rates.png")
    plt.close()

# Create a figure showing trial-averaged responses across all units
def plot_trial_averaged_responses():
    window = (-0.5, 1.0)  # Time window around trial onset
    bin_width = 0.025  # 25ms bins
    
    # Create bins for the histogram
    bins = np.arange(window[0], window[1] + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width/2
    num_bins = len(bin_centers)
    
    # Arrays to store average responses
    rsu_responses = np.zeros((num_rsu, num_bins))
    fsu_responses = np.zeros((num_fsu, num_bins))
    
    # Calculate trial-averaged response for each unit
    rsu_index = 0
    fsu_index = 0
    
    for unit_index in range(num_units):
        spike_times = units["spike_times"][unit_index]
        cell_type = cell_types[unit_index]
        
        # Calculate histogram for this unit
        hist_data = np.zeros(num_bins)
        
        for trial_start in trial_start_times:
            # Find spikes within the window of this trial
            trial_window_start = trial_start + window[0]
            trial_window_end = trial_start + window[1]
            trial_spikes = spike_times[(spike_times >= trial_window_start) & 
                                       (spike_times <= trial_window_end)]
            
            # Convert spike times to times relative to trial start
            relative_spike_times = trial_spikes - trial_start
            
            # Count spikes in each bin
            hist, _ = np.histogram(relative_spike_times, bins=bins)
            hist_data += hist
        
        # Convert to firing rate (spikes/s)
        firing_rate = hist_data / (len(trial_start_times) * bin_width)
        
        # Store in appropriate array
        if cell_type == 1:  # RSU
            rsu_responses[rsu_index] = firing_rate
            rsu_index += 1
        elif cell_type == 2:  # FSU
            fsu_responses[fsu_index] = firing_rate
            fsu_index += 1
    
    # Calculate the mean and SEM across units
    mean_rsu_response = np.mean(rsu_responses, axis=0)
    sem_rsu_response = np.std(rsu_responses, axis=0) / np.sqrt(num_rsu)
    
    mean_fsu_response = np.mean(fsu_responses, axis=0)
    sem_fsu_response = np.std(fsu_responses, axis=0) / np.sqrt(num_fsu)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    plt.plot(bin_centers, mean_rsu_response, color='blue', label='RSU')
    plt.fill_between(bin_centers, 
                     mean_rsu_response - sem_rsu_response,
                     mean_rsu_response + sem_rsu_response,
                     color='blue', alpha=0.3)
    
    plt.plot(bin_centers, mean_fsu_response, color='orange', label='FSU')
    plt.fill_between(bin_centers, 
                     mean_fsu_response - sem_fsu_response,
                     mean_fsu_response + sem_fsu_response,
                     color='orange', alpha=0.3)
    
    # Add a vertical line at t=0 (trial onset)
    plt.axvline(x=0, color='red', linestyle='--', label='Trial Onset')
    
    # Add a vertical line at the mean trial duration
    mean_duration = np.mean(trial_stop_times - trial_start_times)
    plt.axvline(x=mean_duration, color='green', linestyle='--', label='Trial Offset (Mean)')
    
    plt.xlabel('Time relative to trial onset (s)')
    plt.ylabel('Firing Rate (spikes/s)')
    plt.title('Average Response By Cell Type')
    plt.legend()
    plt.xlim(window)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("tmp_scripts/average_response_by_cell_type.png")
    plt.close()

# Select a few example units to create raster and PSTH plots
# Try to pick units that have good spike counts and represent different cell types
def select_example_units(n=3):
    # Calculate the total number of spikes for each unit
    spike_counts = np.array([len(units["spike_times"][i]) for i in range(num_units)])
    
    # Separate by cell type
    rsu_indices = np.where(cell_types == 1)[0]
    fsu_indices = np.where(cell_types == 2)[0]
    
    rsu_spike_counts = spike_counts[rsu_indices]
    fsu_spike_counts = spike_counts[fsu_indices]
    
    # Sort by spike count (descending)
    sorted_rsu_indices = rsu_indices[np.argsort(-rsu_spike_counts)]
    sorted_fsu_indices = fsu_indices[np.argsort(-fsu_spike_counts)]
    
    # Select top units from each type, up to n total
    example_indices = []
    for i in range(min(n//2 + n%2, len(sorted_rsu_indices))):
        example_indices.append(sorted_rsu_indices[i])
    for i in range(min(n//2, len(sorted_fsu_indices))):
        example_indices.append(sorted_fsu_indices[i])
    
    return example_indices[:n]  # Limit to n units

# Execute analyses
print("Selecting example units...")
example_units = select_example_units(3)

print("Creating raster plots and PSTHs for example units...")
for unit_index in example_units:
    create_raster_plot(unit_index)
    create_psth(unit_index)

print("Calculating and plotting average firing rates...")
plot_average_firing_rates()

print("Calculating and plotting trial-averaged responses...")
plot_trial_averaged_responses()

print("Analysis completed!")