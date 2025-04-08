'''
This script compares neural responses across different PRF (pulse repetition frequency) values.
It will:
1. Identify NWB files with different PRF values for the same subject
2. Calculate average neural responses for each PRF
3. Generate plots comparing responses across different PRF values
'''

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import re
from dandi.dandiapi import DandiAPIClient

# Set up plotting to save to file instead of displaying
plt.ioff()  # Turn interactive mode off

print("Getting Dandiset assets...")
client = DandiAPIClient()
dandiset = client.get_dandiset("000945")
assets = list(dandiset.get_assets())

# Extract PRF values from file identifiers (NWB identifiers) when available
def extract_prf(identifier):
    # Try to extract PRF from identifiers like BH498_3000_200
    match = re.search(r'_(\d+)_', identifier)
    if match:
        try:
            prf = int(match.group(1))
            # Verify it's one of the expected PRF values
            if prf in [30, 300, 1500, 3000, 4500]:
                return prf
        except ValueError:
            pass
    return None

# Group assets by subject
subjects = {}
for asset in assets:
    path = asset.path
    subject_id = path.split('/')[0].replace('sub-', '')
    
    if subject_id not in subjects:
        subjects[subject_id] = []
    
    subjects[subject_id].append(asset)

# Choose a subject with multiple PRF values
print("Finding a subject with multiple PRF files...")
chosen_subject = None
subject_files = []
prf_values = []

for subject_id, subject_assets in subjects.items():
    if len(subject_assets) >= 4:  # Look for subjects with at least 4 files
        # Check if we can get PRF values for at least 3 files
        prfs = []
        files_with_prf = []
        
        for asset in subject_assets:
            # Sample a few files to check if we can extract PRF values
            lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/000945/assets/{asset.identifier}/nwb.lindi.json"
            try:
                f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
                nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
                prf = extract_prf(nwb.identifier)
                
                if prf is not None:
                    prfs.append(prf)
                    files_with_prf.append((asset, prf, nwb.identifier))
                
                # Close the file to free resources
                del nwb
                f.close()
                
            except Exception as e:
                print(f"Error opening {asset.path}: {e}")
                continue
        
        # If we found at least 3 different PRF values, use this subject
        unique_prfs = set(prfs)
        if len(unique_prfs) >= 3:
            chosen_subject = subject_id
            subject_files = files_with_prf
            prf_values = sorted(list(unique_prfs))
            break

if chosen_subject is None:
    print("Could not find a suitable subject with multiple PRF values. Using first subject.")
    chosen_subject = list(subjects.keys())[0]
    subject_files = [(asset, None, None) for asset in subjects[chosen_subject][:3]]

print(f"Selected subject: {chosen_subject}")
print(f"PRF values found: {prf_values}")
print(f"Number of files: {len(subject_files)}")

# Function to calculate average firing rates around trial onset
def calculate_avg_firing_rates(nwb, window=(-0.3, 0.5)):
    # Get trials and units
    trials = nwb.intervals["trials"]
    trial_start_times = trials["start_time"][:]
    
    units = nwb.units
    num_units = len(units)
    
    # Get cell type labels
    cell_types = units["celltype_label"].data[:]
    
    # Calculate average firing rate across all units in the defined window
    rsu_rates = []
    fsu_rates = []
    
    for unit_index in range(num_units):
        spike_times = units["spike_times"][unit_index]
        cell_type = cell_types[unit_index]
        
        # Calculate average firing rate around trial onset
        rate_sum = 0
        for trial_start in trial_start_times:
            # Count spikes in the window
            window_start = trial_start + window[0]
            window_end = trial_start + window[1]
            spikes_in_window = np.sum((spike_times >= window_start) & (spike_times < window_end))
            rate = spikes_in_window / (window[1] - window[0])
            rate_sum += rate
        
        # Average across trials
        avg_rate = rate_sum / len(trial_start_times)
        
        # Store based on cell type
        if cell_type == 1:  # RSU
            rsu_rates.append(avg_rate)
        elif cell_type == 2:  # FSU
            fsu_rates.append(avg_rate)
    
    return {
        'rsu_mean': np.mean(rsu_rates) if rsu_rates else 0,
        'rsu_sem': np.std(rsu_rates) / np.sqrt(len(rsu_rates)) if rsu_rates else 0,
        'fsu_mean': np.mean(fsu_rates) if fsu_rates else 0,
        'fsu_sem': np.std(fsu_rates) / np.sqrt(len(fsu_rates)) if fsu_rates else 0,
        'rsu_count': len(rsu_rates),
        'fsu_count': len(fsu_rates)
    }

# Calculate PSTH for different PRFs
def calculate_psth_by_prf(nwb, window=(-0.3, 1.0), bin_width=0.05):
    # Get trials and units
    trials = nwb.intervals["trials"]
    trial_start_times = trials["start_time"][:]
    
    units = nwb.units
    num_units = len(units)
    
    # Get cell type labels
    cell_types = units["celltype_label"].data[:]
    
    # Create bins for the histogram
    bins = np.arange(window[0], window[1] + bin_width, bin_width)
    bin_centers = bins[:-1] + bin_width/2
    num_bins = len(bin_centers)
    
    # Arrays to store average responses
    rsu_responses = np.zeros((num_bins,))
    fsu_responses = np.zeros((num_bins,))
    rsu_count = 0
    fsu_count = 0
    
    # Calculate histogram for each unit
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
        
        # Add to the appropriate cell type average
        if cell_type == 1:  # RSU
            rsu_responses += firing_rate
            rsu_count += 1
        elif cell_type == 2:  # FSU
            fsu_responses += firing_rate
            fsu_count += 1
    
    # Calculate averages
    if rsu_count > 0:
        rsu_responses /= rsu_count
    if fsu_count > 0:
        fsu_responses /= fsu_count
    
    return bin_centers, rsu_responses, fsu_responses, rsu_count, fsu_count

# Process the selected files
prf_results = {}
psth_results = {}

for asset, prf, identifier in subject_files:
    if prf is None:
        continue
        
    print(f"Processing file with PRF {prf}Hz...")
    
    lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/000945/assets/{asset.identifier}/nwb.lindi.json"
    try:
        f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
        nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
        
        # Calculate average firing rates
        prf_results[prf] = calculate_avg_firing_rates(nwb)
        
        # Calculate PSTH
        bin_centers, rsu_responses, fsu_responses, rsu_count, fsu_count = calculate_psth_by_prf(nwb)
        psth_results[prf] = {
            'bin_centers': bin_centers,
            'rsu_responses': rsu_responses,
            'fsu_responses': fsu_responses,
            'rsu_count': rsu_count,
            'fsu_count': fsu_count
        }
        
        # Clean up
        del nwb
        f.close()
        
    except Exception as e:
        print(f"Error processing file with PRF {prf}Hz: {e}")

# Plot average firing rate vs PRF
if prf_results:
    prfs = sorted(prf_results.keys())
    rsu_means = [prf_results[prf]['rsu_mean'] for prf in prfs]
    rsu_sems = [prf_results[prf]['rsu_sem'] for prf in prfs]
    fsu_means = [prf_results[prf]['fsu_mean'] for prf in prfs]
    fsu_sems = [prf_results[prf]['fsu_sem'] for prf in prfs]
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(prfs, rsu_means, yerr=rsu_sems, marker='o', linestyle='-', label='RSU', color='blue')
    plt.errorbar(prfs, fsu_means, yerr=fsu_sems, marker='s', linestyle='-', label='FSU', color='orange')
    
    plt.xlabel('PRF (Hz)')
    plt.ylabel('Average Firing Rate (spikes/s)')
    plt.title(f'Effect of PRF on Firing Rate (Subject {chosen_subject})')
    plt.xscale('log')  # Log scale for PRF since values span orders of magnitude
    plt.xticks(prfs, [str(prf) for prf in prfs])  # Show actual PRF values
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tmp_scripts/firing_rate_vs_prf.png")
    plt.close()
    
    # Print unit counts for reference
    print("\nUnit counts for each PRF:")
    for prf in prfs:
        print(f"PRF {prf}Hz: {prf_results[prf]['rsu_count']} RSUs, {prf_results[prf]['fsu_count']} FSUs")

# Plot PSTH for different PRFs (RSU)
if psth_results:
    plt.figure(figsize=(12, 8))
    
    for prf in sorted(psth_results.keys()):
        bin_centers = psth_results[prf]['bin_centers']
        rsu_responses = psth_results[prf]['rsu_responses']
        
        plt.plot(bin_centers, rsu_responses, label=f'PRF {prf}Hz')
    
    # Add a vertical line at t=0 (trial onset)
    plt.axvline(x=0, color='red', linestyle='--', label='Trial Onset')
    
    # Add a vertical line at t=0.067 (mean trial duration)
    plt.axvline(x=0.067, color='green', linestyle='--', label='Stimulation End')
    
    plt.xlabel('Time relative to trial onset (s)')
    plt.ylabel('Firing Rate (spikes/s)')
    plt.title(f'RSU Responses at Different PRFs (Subject {chosen_subject})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("tmp_scripts/rsu_response_by_prf.png")
    plt.close()
    
    # Plot PSTH for different PRFs (FSU)
    plt.figure(figsize=(12, 8))
    
    for prf in sorted(psth_results.keys()):
        bin_centers = psth_results[prf]['bin_centers']
        fsu_responses = psth_results[prf]['fsu_responses']
        
        plt.plot(bin_centers, fsu_responses, label=f'PRF {prf}Hz')
    
    # Add a vertical line at t=0 (trial onset)
    plt.axvline(x=0, color='red', linestyle='--', label='Trial Onset')
    
    # Add a vertical line at t=0.067 (mean trial duration)
    plt.axvline(x=0.067, color='green', linestyle='--', label='Stimulation End')
    
    plt.xlabel('Time relative to trial onset (s)')
    plt.ylabel('Firing Rate (spikes/s)')
    plt.title(f'FSU Responses at Different PRFs (Subject {chosen_subject})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("tmp_scripts/fsu_response_by_prf.png")
    plt.close()

print("Analysis completed!")