# Basic exploratory script for Dandiset 000945 NWB file
# Loads the NWB file, prints neuron/unit info, trial counts, electrode metadata, and some example spike times

import pynwb
import h5py
import remfile
import numpy as np

url = "https://api.dandiarchive.org/api/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/download/"

file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

print("Session description:", nwb.session_description)
print("Subject ID:", nwb.subject.subject_id, "; Age:", nwb.subject.age, "; Sex:", nwb.subject.sex)
print("Institution:", nwb.institution)

# Trials info
print("Number of trials:", len(nwb.trials.id))

# Electrode info
print("Number of electrodes:", len(nwb.electrodes.id))
print("Electrode columns:", nwb.electrodes.colnames)

# Units info
print("Number of units (neurons):", len(nwb.units.id))
print("Unit columns:", nwb.units.colnames)

# Cell type label distribution
if 'celltype_label' in nwb.units.colnames:
    labels = nwb.units['celltype_label'][:]
    unique, counts = np.unique(labels, return_counts=True)
    print("Cell types in 'celltype_label':")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")
else:
    print("No 'celltype_label' found.")

# Example spike times for first 5 units
print("Example spike times for first 5 units:")
for idx in range(min(5, len(nwb.units.id))):
    uid = nwb.units.id[idx]
    spikes = nwb.units['spike_times'][idx]
    print(f"  Unit ID {uid} first 10 spikes:", spikes[:10])

io.close()