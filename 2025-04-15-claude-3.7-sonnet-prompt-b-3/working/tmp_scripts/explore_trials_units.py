"""
This script explores the trial structure and unit activity in the NWB file.
We want to understand:
1. The timing of the trials (how they're distributed)
2. The types of neurons recorded
3. The basic activity patterns of the neurons
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load
url = "https://api.dandiarchive.org/api/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

print("NWB File Information:")
print(f"Subject: {nwb.subject.subject_id}")
print(f"Session: {nwb.session_description}")
print(f"Species: {nwb.subject.species}")
print(f"Age: {nwb.subject.age}")
print(f"Sex: {nwb.subject.sex}")

# Examine trials
trials_df = nwb.trials.to_dataframe()
print("\nTrials information:")
print(f"Number of trials: {len(trials_df)}")
print("First 5 trials:")
print(trials_df.head())

# Calculate trial durations and inter-trial intervals
trials_df['duration'] = trials_df['stop_time'] - trials_df['start_time']
trials_df['iti'] = trials_df['start_time'].shift(-1) - trials_df['stop_time']

print("\nTrial statistics:")
print(f"Mean trial duration: {trials_df['duration'].mean():.3f} seconds")
print(f"Mean inter-trial interval: {trials_df['iti'].dropna().mean():.3f} seconds")

# Examine units
units_df = nwb.units.to_dataframe()
print("\nUnits information:")
print(f"Number of units: {len(units_df)}")
print("Cell types:")
print(units_df['celltype_label'].value_counts())

# Plot trial start times to see their distribution
plt.figure(figsize=(10, 4))
plt.plot(trials_df.index, trials_df['start_time'], 'o-')
plt.title('Trial Start Times')
plt.xlabel('Trial Number')
plt.ylabel('Time (s)')
plt.grid(True)
plt.savefig('tmp_scripts/trial_start_times.png')

# Plot trial durations
plt.figure(figsize=(10, 4))
plt.hist(trials_df['duration'], bins=30)
plt.title('Trial Duration Distribution')
plt.xlabel('Duration (s)')
plt.ylabel('Count')
plt.grid(True)
plt.savefig('tmp_scripts/trial_durations.png')

# Plot the number of spikes per unit
spike_counts = []
for i, row in units_df.iterrows():
    spike_counts.append(len(row['spike_times']))

plt.figure(figsize=(10, 4))
plt.bar(range(len(spike_counts)), spike_counts)
plt.title('Number of Spikes per Unit')
plt.xlabel('Unit ID')
plt.ylabel('Spike Count')
plt.grid(True)
plt.savefig('tmp_scripts/spike_counts.png')

# Plot spike counts by cell type
cell_types = units_df['celltype_label'].unique()
cell_type_counts = {ct: [] for ct in cell_types}

for i, row in units_df.iterrows():
    cell_type = row['celltype_label']
    spike_count = len(row['spike_times'])
    cell_type_counts[cell_type].append(spike_count)

plt.figure(figsize=(10, 5))
for i, cell_type in enumerate(cell_types):
    plt.boxplot(cell_type_counts[cell_type], positions=[i], widths=0.6)
plt.title('Spike Counts by Cell Type')
plt.xlabel('Cell Type')
plt.ylabel('Spike Count')
plt.xticks(range(len(cell_types)), cell_types)
plt.grid(True)
plt.savefig('tmp_scripts/spike_counts_by_cell_type.png')

# Get spike times for first 5 units to examine their raster
plt.figure(figsize=(12, 8))
for i in range(5):  # First 5 units
    unit_id = units_df.index[i]
    spike_times = units_df.loc[unit_id, 'spike_times']
    cell_type = units_df.loc[unit_id, 'celltype_label']
    
    # Plot spike times as a raster
    plt.subplot(5, 1, i+1)
    plt.eventplot(spike_times, lineoffsets=0, linelengths=0.5)
    plt.title(f'Unit {unit_id} - {cell_type}')
    plt.xlabel('Time (s)')
    plt.ylabel('Spikes')
    plt.xlim(0, 100)  # First 100 seconds

plt.tight_layout()
plt.savefig('tmp_scripts/spike_rasters.png')

print("Exploration completed and plots saved.")