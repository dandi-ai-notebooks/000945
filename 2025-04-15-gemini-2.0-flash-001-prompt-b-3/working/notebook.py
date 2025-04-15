# %% [markdown]
# # Exploring Dandiset 000945: Neural Spiking Data in Awake Rat Somatosensory Cortex

# %% [markdown]
# **Disclaimer:** This notebook was AI-generated using dandi-notebook-gen and has not been fully verified. Use caution when interpreting the code or results.

# %% [markdown]
# ## Overview of the Dandiset

# %% [markdown]
# This Dandiset (000945) contains neural spiking data recorded from the somatosensory cortex of awake rats in response to transcranial focused ultrasound stimulation (tFUS). The data includes recordings from multiple rats and different pulse repetition frequencies (PRFs) of ultrasound stimulation.

# %% [markdown]
# ## What this notebook covers

# %% [markdown]
# This notebook demonstrates how to load and visualize data from this Dandiset, focusing on spike times and cell type labels. We will load data from one of the NWB files in the Dandiset and generate plots to visualize spike times and their distributions.

# %% [markdown]
# ## Required Packages

# %% [markdown]
# The following packages are required to run this notebook:
# - pynwb
# - h5py
# - remfile
# - matplotlib
# - numpy
# - seaborn

# %%
from dandi.dandiapi import DandiAPIClient
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme()

# %% [markdown]
# ## Loading the Dandiset

# %% [markdown]
# First, we connect to the DANDI archive and load the Dandiset.

# %%
client = DandiAPIClient()
dandiset = client.get_dandiset("000945")
assets = list(dandiset.get_assets())

print(f"Found {len(assets)} assets in the dataset")
print("\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Loading Data from an NWB File

# %% [markdown]
# We will load data from the first NWB file in the Dandiset: `sub-BH497/sub-BH497_ses-20240310T143729_ecephys.nwb`. This file contains neural spiking data recorded from rat BH497 on March 10, 2024.

# %%
url = "https://api.dandiarchive.org/api/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# %% [markdown]
# Now we can inspect the contents of the NWB file.

# %%
nwb.session_description # (str) Awake S1 Stimulation by tFUS
nwb.identifier # (str) BH498_3000_200_anes
nwb.session_start_time # (datetime) 2024-03-10T14:37:29-04:00
nwb.electrode_groups["shank1"] # (ElectrodeGroup)

# %% [markdown]
# We can also get information about the electrodes.

# %%
nwb.electrodes.colnames

# %% [markdown]
# ## Visualizing Spike Times

# %% [markdown]
# Next, we will load the spike times for a subset of units and visualize them.

# %%
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
plt.show()

# %% [markdown]
# The plot above shows the spike times for the selected units, colored by cell type.

# %% [markdown]
# ## Distribution of Spike Times

# %% [markdown]
# Now, we will plot the distribution of spike times for each cell type.

# %%
# Plot distribution of spike times
plt.figure(figsize=(10, 6))
sns.histplot(x=spike_times, hue=celltype_labels, kde=True)
plt.xlabel("Time (s)")
plt.ylabel("Number of Spikes")
plt.title("Distribution of Spike Times for Selected Units")
plt.show()

# %% [markdown]
# This histogram shows the distribution of spike times for the selected units.

# %% [markdown]
# ## Summary and Future Directions

# %% [markdown]
# This notebook demonstrated how to load and visualize neural spiking data from Dandiset 000945. We loaded data from an NWB file, extracted spike times and cell type labels, and generated plots to visualize the data.

# %% [markdown]
# Future directions for analysis could include:
# - Analyzing spike rates and firing patterns for different PRFs of ultrasound stimulation.
# - Comparing neural activity between different subjects or experimental conditions.
# - Performing more advanced signal processing techniques to extract features from the neural data.