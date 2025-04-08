"""
Analysis of trial structure and temporal patterns in the neural data:
1. Distribution of inter-trial intervals
2. PSTH (Peri-Stimulus Time Histogram) across all units
3. Trial duration consistency check
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set up seaborn style
sns.set_theme()

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000945/assets/e35653b4-0a0b-41bf-bf71-0c37e0d96509/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Get trials info
trials = nwb.intervals["trials"]
trial_starts = trials["start_time"][:]
trial_stops = trials["stop_time"][:]

# Calculate inter-trial intervals
itis = trial_starts[1:] - trial_starts[:-1]

# Plot 1: Distribution of inter-trial intervals
plt.figure(figsize=(8, 6))
plt.hist(itis, bins=30)
plt.xlabel('Inter-Trial Interval (s)')
plt.ylabel('Count')
plt.title('Distribution of Inter-Trial Intervals')
plt.savefig('tmp_scripts/iti_distribution.png')
plt.close()

# Create PSTH
window = [-0.5, 1.0]  # Same window as before
bin_size = 0.05  # 50ms bins
bins = np.arange(window[0], window[1] + bin_size, bin_size)
bin_centers = bins[:-1] + bin_size/2
all_unit_psth = np.zeros((len(nwb.units), len(bins)-1))

# Calculate PSTH for each unit
for unit_idx in range(len(nwb.units)):
    spike_times = nwb.units["spike_times"][unit_idx]
    unit_hist = np.zeros((len(trial_starts), len(bins)-1))
    
    for trial_idx, trial_start in enumerate(trial_starts):
        # Get spikes in window around this trial
        mask = (spike_times >= trial_start + window[0]) & (spike_times <= trial_start + window[1])
        trial_spikes = spike_times[mask] - trial_start
        
        # Create histogram for this trial
        hist, _ = np.histogram(trial_spikes, bins=bins)
        unit_hist[trial_idx] = hist
    
    # Average across trials and convert to rate
    all_unit_psth[unit_idx] = np.mean(unit_hist, axis=0) / bin_size

# Plot 2: Average PSTH across all units
plt.figure(figsize=(10, 6))
mean_psth = np.mean(all_unit_psth, axis=0)
sem_psth = np.std(all_unit_psth, axis=0) / np.sqrt(len(nwb.units))

plt.plot(bin_centers, mean_psth, 'b-', label='Mean')
plt.fill_between(bin_centers, 
                 mean_psth - sem_psth,
                 mean_psth + sem_psth,
                 alpha=0.3)
plt.axvline(x=0, color='r', linestyle='--', label='Stimulation')
plt.xlabel('Time relative to stimulation (s)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Population PSTH')
plt.legend()
plt.savefig('tmp_scripts/population_psth.png')
plt.close()

# Print some statistics about trials
print(f"Number of trials: {len(trial_starts)}")
print(f"\nInter-trial interval statistics:")
print(f"Mean ITI: {np.mean(itis):.3f} s")
print(f"Median ITI: {np.median(itis):.3f} s")
print(f"STD of ITI: {np.std(itis):.3f} s")
print(f"Min ITI: {np.min(itis):.3f} s")
print(f"Max ITI: {np.max(itis):.3f} s")

print(f"\nTrial duration statistics:")
durations = trial_stops - trial_starts
print(f"Mean duration: {np.mean(durations):.3f} s")
print(f"Median duration: {np.median(durations):.3f} s")
print(f"STD of duration: {np.std(durations):.3f} s")