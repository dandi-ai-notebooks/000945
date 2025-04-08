# %% [markdown]
# # Analysis of Neural Responses to Transcranial Focused Ultrasound Stimulation
# 
# **Note: This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Please be cautious when interpreting the code or results.**
#
# This notebook demonstrates how to load and analyze neural spiking data from DANDI:000945, which contains recordings from the awake rat somatosensory cortex during transcranial focused ultrasound stimulation (tFUS).
#
# ## Dataset Overview
#
# This dataset includes neural recordings from the somatosensory cortex of awake head-fixed rats during trials of transcranial focused ultrasound stimulation. Key details:
#
# - Recordings were made using 32-channel NeuroNexus electrodes
# - Different pulse repetition frequencies (PRFs) were tested
# - Each recording contains 500 trials
# - Ultrasound stimulation was delivered every ~2.5 seconds with 10% jitter
# - Multiple subjects and conditions are available in the dataset
#
# ## Required Packages
#
# To run this notebook, you'll need the following packages installed:
# - pynwb
# - lindi
# - numpy
# - matplotlib
# - seaborn

# %%
import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the plotting style
sns.set_theme()

# %% [markdown]
# ## Loading the Dataset
#
# First, let's use the DANDI API to get a list of all assets in this Dandiset:

# %%
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
dandiset = client.get_dandiset("000945")
assets = list(dandiset.get_assets())
print(f"Total number of assets: {len(assets)}")

# %% [markdown]
# For this demonstration, we'll analyze one recording session. We'll use a file that contains
# neural responses to 300 Hz PRF stimulation:

# %%
# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000945/assets/e35653b4-0a0b-41bf-bf71-0c37e0d96509/nwb.lindi.json")
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic session info
print(f"Session ID: {nwb.identifier}")
print(f"Session description: {nwb.session_description}")
print(f"Institution: {nwb.institution}")
print(f"\nSubject information:")
print(f"Species: {nwb.subject.species}")
print(f"Sex: {nwb.subject.sex}")
print(f"Age: {nwb.subject.age}")

# %% [markdown]
# ## Understanding the Trial Structure
#
# The experiment consists of repeated trials of ultrasound stimulation. Let's examine the 
# trial timing and structure:

# %%
# Get trials information
trials = nwb.intervals["trials"]
trial_starts = trials["start_time"][:]
trial_stops = trials["stop_time"][:]

# Calculate inter-trial intervals
itis = trial_starts[1:] - trial_starts[:-1]

# Plot distribution of inter-trial intervals
plt.figure(figsize=(10, 6))
plt.hist(itis, bins=30)
plt.xlabel('Inter-Trial Interval (s)')
plt.ylabel('Count')
plt.title('Distribution of Inter-Trial Intervals')
plt.show()

# Print trial timing statistics
print("\nTrial timing statistics:")
print(f"Number of trials: {len(trial_starts)}")
print(f"Mean inter-trial interval: {np.mean(itis):.3f} s")
print(f"Trial duration: {np.mean(trial_stops - trial_starts):.3f} s")

# %% [markdown]
# The trials are spaced approximately 2.5 seconds apart with some jitter, as specified in the
# dataset description. Each trial has a consistent duration of about 2.2 seconds.

# %% [markdown]
# ## Exploring Neural Activity
#
# The dataset contains spike times from multiple units recorded during the experiment. Let's look
# at the basic properties of these neural recordings:

# %%
# Get units information
units = nwb.units
unit_ids = units["id"].data
num_units = len(unit_ids)

# Calculate firing rates
recording_duration = trial_stops[-1] - trial_starts[0]
firing_rates = []

for unit_idx in range(num_units):
    spike_times = units["spike_times"][unit_idx]
    firing_rates.append(len(spike_times) / recording_duration)

# Plot distribution of firing rates
plt.figure(figsize=(10, 6))
plt.hist(firing_rates, bins=20)
plt.xlabel('Firing Rate (Hz)')
plt.ylabel('Number of Units')
plt.title('Distribution of Firing Rates')
plt.show()

print(f"\nFiring rate statistics:")
print(f"Number of units: {num_units}")
print(f"Mean firing rate: {np.mean(firing_rates):.2f} Hz")
print(f"Median firing rate: {np.median(firing_rates):.2f} Hz")
print(f"Range: {np.min(firing_rates):.2f} - {np.max(firing_rates):.2f} Hz")

# %% [markdown]
# The units show a range of firing rates, with most units firing between 10-30 Hz. This
# variation in firing rates is typical for cortical recordings.

# %% [markdown]
# ## Analyzing Neural Responses to Stimulation
#
# To examine how neurons respond to the ultrasound stimulation, we can create a peri-stimulus
# time histogram (PSTH) that shows the average firing rate around the time of stimulation:

# %%
# Parameters for PSTH
window = [-0.5, 1.0]  # Time window around stimulation (in seconds)
bin_size = 0.05      # 50ms bins
bins = np.arange(window[0], window[1] + bin_size, bin_size)
bin_centers = bins[:-1] + bin_size/2

# Calculate PSTH for all units
all_unit_psth = np.zeros((len(units), len(bins)-1))

for unit_idx in range(len(units)):
    spike_times = units["spike_times"][unit_idx]
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

# Plot population PSTH
plt.figure(figsize=(12, 6))
mean_psth = np.mean(all_unit_psth, axis=0)
sem_psth = np.std(all_unit_psth, axis=0) / np.sqrt(len(units))

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
plt.show()

# %% [markdown]
# ## Next Steps
#
# This notebook demonstrates basic loading and analysis of the neural data from this dataset. 
# Researchers might want to explore:
#
# 1. Comparing responses across different PRF conditions
# 2. Analyzing individual unit responses in more detail
# 3. Investigating response latencies
# 4. Comparing awake vs. anesthetized conditions
# 5. Looking at spatial patterns across the recording array
#
# The dataset contains multiple subjects and conditions, allowing for comprehensive analysis
# of how transcranial focused ultrasound affects neural activity in the somatosensory cortex.