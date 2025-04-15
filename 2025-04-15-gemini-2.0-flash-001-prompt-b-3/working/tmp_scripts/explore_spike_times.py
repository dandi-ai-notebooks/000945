import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Select a random subset of units (e.g., 10 units)
unit_ids = nwb.units.id[:]
np.random.shuffle(unit_ids)
selected_unit_ids = unit_ids[:10]

# Collect spike times and cell type labels for the selected units
spike_times = []
celltype_labels = []
for unit_id in selected_unit_ids:
    unit_index = np.where(nwb.units.id[:] == unit_id)[0][0]
    times = nwb.units['spike_times'][unit_index]
    # Limit spike times to the first 10 seconds
    times = times[times < 10]
    spike_times.extend(times)
    celltype_labels.extend([nwb.units['celltype_label'][unit_index]] * len(times))

# Plot spike times
plt.figure(figsize=(10, 6))
sns.scatterplot(x=spike_times, y=np.arange(len(spike_times)), hue=celltype_labels, s=5)
plt.xlabel("Time (s)")
plt.ylabel("Spike Number")
plt.title("Spike Times for Selected Units")
plt.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig("tmp_scripts/spike_times_scatter.png")
plt.close()

# Plot distribution of spike times
plt.figure(figsize=(10, 6))
sns.histplot(x=spike_times, hue=celltype_labels, kde=True)
plt.xlabel("Time (s)")
plt.ylabel("Number of Spikes")
plt.title("Distribution of Spike Times for Selected Units")
plt.savefig("tmp_scripts/spike_times_hist.png")
plt.close()