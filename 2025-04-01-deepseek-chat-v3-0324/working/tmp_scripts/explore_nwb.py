"""
First exploratory script for NWB file analysis
Goals:
1. Print basic file and subject metadata
2. Show trial structure information
3. Print electrode positions and properties
4. Show overview of unit (neuron) data
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the NWB file
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/000945/assets/a7549e3f-9b14-432a-be65-adb5f6811343/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

print("=== NWB File Information ===")
print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject species: {nwb.subject.species}")
print(f"Subject sex: {nwb.subject.sex}")
print(f"Subject age: {nwb.subject.age}")

print("\n=== Trial Information ===")
trials = nwb.intervals["trials"]
print(f"Number of trials: {len(trials['start_time'][:])}")
print(f"First trial start time: {trials['start_time'][0]}")
print(f"First trial end time: {trials['stop_time'][0]}")
print(f"Trial duration: {trials['stop_time'][0] - trials['start_time'][0]} seconds")

print("\n=== Electrode Information ===")
electrodes = nwb.electrodes
print(f"Number of electrodes: {len(electrodes['id'][:])}")
print("Electrode positions (first 5):")
for i in range(5):
    x = electrodes["x"][i]
    y = electrodes["y"][i] 
    z = electrodes["z"][i]
    print(f"Electrode {i+1}: x={x}, y={y}, z={z}")

print("\n=== Unit (Neuron) Information ===")
units = nwb.units
unit_ids = units["id"][:]
cell_types = units["celltype_label"][:]
print(f"Number of units: {len(unit_ids)}")
print(f"Cell type labels (1=RSU, 2=FSU):")
unique, counts = np.unique(cell_types, return_counts=True)
for val, count in zip(unique, counts):
    print(f"Type {int(val)}: {count} units")

# Plot firing rates across all units
spike_counts = [len(units["spike_times"][i]) for i in range(len(unit_ids))]

plt.figure(figsize=(10, 5))
plt.bar(range(len(spike_counts)), spike_counts)
plt.xlabel("Unit Index")
plt.ylabel("Spike Count")
plt.title("Spike Counts Across Units")
plt.savefig("tmp_scripts/unit_spike_counts.png")
