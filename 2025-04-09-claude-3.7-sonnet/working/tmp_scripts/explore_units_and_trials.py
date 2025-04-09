# This script explores the units (neural recordings) and trials data 
# in the NWB file to understand their structure and content

import pynwb
import h5py
import remfile
import numpy as np

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/655fe6cf-a152-412b-9d20-71c6db670629/download/"
file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

# Print basic information
print(f"NWB File: {nwb.identifier}")
print(f"Session Description: {nwb.session_description}")
print(f"Subject: {nwb.subject.subject_id} ({nwb.subject.sex}, Age: {nwb.subject.age})")
print(f"Species: {nwb.subject.species}")
print(f"Institution: {nwb.institution}")
print()

# Explore units (neurons)
print("=== UNITS INFO ===")
print(f"Number of units: {len(nwb.units.id[:])}")
print(f"Unit columns: {nwb.units.colnames}")

# Get unique cell types
if 'celltype_label' in nwb.units.colnames:
    cell_types = set()
    for i in range(len(nwb.units.id[:])):
        cell_type = nwb.units['celltype_label'][i]
        if cell_type:  # Check if not empty
            cell_types.add(cell_type)
    print(f"Cell types: {cell_types}")

# Sample some spike times
print("\nSample spike times for first 3 units:")
for i in range(min(3, len(nwb.units.id[:]))):
    spike_times = nwb.units['spike_times'][i]
    n_spikes = len(spike_times[:])
    print(f"Unit {i} (ID: {nwb.units.id[i]}): {n_spikes} spikes")
    if n_spikes > 0:
        print(f"  First 5 spike times: {spike_times[:5]}")
        print(f"  Mean firing rate: {n_spikes / (nwb.trials['stop_time'][-1] - nwb.trials['start_time'][0]):.2f} Hz")
print()

# Explore trials
print("=== TRIALS INFO ===")
print(f"Number of trials: {len(nwb.trials.id[:])}")
print(f"Trial columns: {nwb.trials.colnames}")

# Sample some trial data
print("\nSample trial timing for first 5 trials:")
for i in range(min(5, len(nwb.trials.id[:]))):
    start = nwb.trials['start_time'][i]
    stop = nwb.trials['stop_time'][i]
    duration = stop - start
    print(f"Trial {i} (ID: {nwb.trials.id[i]}): Start={start:.2f}s, Stop={stop:.2f}s, Duration={duration:.3f}s")

# Calculate inter-trial intervals
if len(nwb.trials.id[:]) > 1:
    itis = []
    for i in range(1, len(nwb.trials.id[:])):
        iti = nwb.trials['start_time'][i] - nwb.trials['start_time'][i-1]
        itis.append(iti)
    
    print(f"\nInter-trial intervals: Mean={np.mean(itis):.3f}s, Min={np.min(itis):.3f}s, Max={np.max(itis):.3f}s")
    print(f"Total recording duration: {nwb.trials['stop_time'][-1] - nwb.trials['start_time'][0]:.2f} seconds")

# Check if there's any electrode info
print("\n=== ELECTRODE INFO ===")
if hasattr(nwb, 'electrodes'):
    print(f"Number of electrodes: {len(nwb.electrodes.id[:])}")
    print(f"Electrode columns: {nwb.electrodes.colnames}")
    
    # Sample some electrode info
    if 'location' in nwb.electrodes.colnames:
        locations = set()
        for i in range(len(nwb.electrodes.id[:])):
            loc = nwb.electrodes['location'][i]
            if loc:  # Check if not empty
                locations.add(loc)
        print(f"Recording locations: {locations}")
else:
    print("No electrode information found.")

# Close the file
io.close()
f.close()
file.close()