import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# This script explores the units data in the NWB file.

url = "https://api.dandiarchive.org/api/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode="r")
nwb = io.read()

# Get the units data
units = nwb.units
spike_times = units.spike_times[:]
celltype_label = units.celltype_label[:]
unit_ids = units.id[:]

# Print the unique cell types
print("Unique cell types:", np.unique(celltype_label))

# Inspect the structure of spike_times
print("Type of spike_times:", type(spike_times))
print("First element of spike_times:", spike_times[0])

# Create a plot of the number of spikes for each unit
num_units_to_plot = 10
num_spikes = []
for i in range(num_units_to_plot):
    spike_times_for_unit = nwb.units['spike_times'][i]
    num_spikes.append(len(spike_times_for_unit))

unit_ids_to_plot = unit_ids[:num_units_to_plot]

plt.figure(figsize=(10, 5))
plt.bar(unit_ids_to_plot, num_spikes)
plt.xlabel("Unit ID")
plt.ylabel("Number of Spikes")
plt.title(f"Number of Spikes per Unit (First {num_units_to_plot} Units)")
plt.savefig("explore/num_spikes_per_unit.png")