"""
This script explores the basic metadata and structure of an NWB file from Dandiset 000945.
It will print information about the file, subject, electrodes, trials, and units.
"""

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/b88188c8-4e4c-494c-8dab-806b1efd55eb/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Print basic file information
print("=== NWB File Information ===")
print(f"Session Description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session Start Time: {nwb.session_start_time}")
print(f"Institution: {nwb.institution}")

# Print subject information
print("\n=== Subject Information ===")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Age: {nwb.subject.age}")
print(f"Sex: {nwb.subject.sex}")
print(f"Species: {nwb.subject.species}")
print(f"Description: {nwb.subject.description}")

# Print electrode information
print("\n=== Electrode Information ===")
print(f"Number of electrodes: {len(nwb.electrodes.id[:])}")
print(f"Electrode columns: {nwb.electrodes.colnames}")

# Get electrodes dataframe
electrodes_df = nwb.electrodes.to_dataframe()
print("\nFirst 5 electrodes:")
print(electrodes_df.head())

# Print trials information
print("\n=== Trials Information ===")
print(f"Number of trials: {len(nwb.trials.id[:])}")
print(f"Trials columns: {nwb.trials.colnames}")

# Get trials dataframe
trials_df = nwb.trials.to_dataframe()
print("\nFirst 5 trials:")
print(trials_df.head())

# Calculate trial durations
trial_durations = trials_df['stop_time'] - trials_df['start_time']
print(f"\nAverage trial duration: {np.mean(trial_durations):.6f} seconds")
print(f"Minimum trial duration: {np.min(trial_durations):.6f} seconds")
print(f"Maximum trial duration: {np.max(trial_durations):.6f} seconds")

# Print units information
print("\n=== Units Information ===")
print(f"Number of units: {len(nwb.units.id[:])}")
print(f"Units columns: {nwb.units.colnames}")

# Get units dataframe
units_df = nwb.units.to_dataframe()
print("\nFirst 5 units:")
print(units_df.head())

# Count cell types
if 'celltype_label' in units_df.columns:
    print("\nCell type distribution:")
    cell_type_counts = units_df['celltype_label'].value_counts()
    for cell_type, count in cell_type_counts.items():
        cell_type_name = "RSU" if cell_type == 1 else "FSU" if cell_type == 2 else f"Unknown ({cell_type})"
        print(f"- {cell_type_name}: {count}")

io.close()
h5_file.close()
remote_file.close()