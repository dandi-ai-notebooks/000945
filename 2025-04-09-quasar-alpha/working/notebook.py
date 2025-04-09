# %% [markdown]
# # Dandiset 000945 Exploration Notebook
# 
# This notebook was **AI-generated using dandi-notebook-gen** and has **not been fully scientifically verified**. Please use caution when executing code or interpreting plots and results.
#
# **Dandiset:** Neural Spiking Data in the Awake Rat Somatosensory Cortex Responding to Trials of Transcranial Focused Ultrasound Stimulation
# 
# **Description:** Recordings from awake head-fixed rats during transcranial focused ultrasound (tFUS) stimulation. Multi-channel chronic recordings using 32-channel electrodes, tested with different ultrasound pulse repetition frequencies (PRFs). Data includes spike times with cell type labels, trial intervals, and metadata about experimental conditions.
# 
# **Dataset Size:** 75 NWB files across multiple subjects and sessions
#
# **Citation:**  
# Ramachandran *et al.* (2025), available at: https://dandiarchive.org/dandiset/000945/draft
#
# ---
# 
# **In this notebook you will learn how to:**
# * Retrieve dataset metadata and assets programmatically
# * Understand the organization of an example recording
# * Access electrode and unit info
# * Visualize neuron population composition
# * Generate a spike raster plot aligned to trials
#
# ---
# 
# ## Setup
# This notebook assumes `pynwb`, `h5py`, `remfile`, `matplotlib`, and `seaborn` are installed.

# %%
from dandi.dandiapi import DandiAPIClient

client = DandiAPIClient()
dandiset = client.get_dandiset("000945")
assets = list(dandiset.get_assets())
print(f"Number of assets in Dandiset: {len(assets)}")

for asset in assets[:5]:
    print(asset.identifier, asset.path, asset.size)

print("...")

# %% [markdown]
# ## Selecting an Example NWB File
# 
# For this demo we'll use a representative 12MB file:
# `/sub-BH497/sub-BH497_ses-20240310T145814_ecephys.nwb`
# accessed via the API download URL.

# %%
import remfile
import h5py
import pynwb

url = "https://api.dandiarchive.org/api/assets/f88a9bec-23d6-4444-8b97-8083e45057c9/download/"

file_obj = remfile.File(url)
f = h5py.File(file_obj)
io = pynwb.NWBHDF5IO(file=f)
nwbfile = io.read()

print("NWB Session:", nwbfile.session_description)
print("Subject ID:", nwbfile.subject.subject_id)
print("Recording start time:", nwbfile.session_start_time)
print("Institution:", nwbfile.institution)

# %% [markdown]
# ## Metadata: Electrodes
# 
# The file contains 32 electrodes grouped under `shank1` targeting somatosensory cortex.

# %%
electrodes_table = nwbfile.electrodes
print("Electrode columns:", electrodes_table.colnames)
print("Number of electrodes:", len(electrodes_table.id))
print(electrodes_table.to_dataframe().head())

# %% [markdown]
# ## Metadata: Units and Cell Types
# 
# The file contains spike times for **64 units**. Two main cell type labels are present, here represented numerically as "1.0" and "2.0".

# %%
units_table = nwbfile.units
print("Unit columns:", units_table.colnames)
print("Number of units:", len(units_table.id))

cell_types = units_table['celltype_label'][:]
cell_types = [ct.decode() if isinstance(ct, bytes) else str(ct) for ct in cell_types]

from collections import Counter
counts = Counter(cell_types)
print("Unit counts by cell type:", counts)

# %% [markdown]
# ### Unit Cell Type Distribution
# 
# Below is a bar chart indicating relative counts of two identified cell types:

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set_theme()

plt.figure(figsize=(8,6))
sns.barplot(x=list(counts.keys()), y=list(counts.values()))
plt.xlabel("Cell Type")
plt.ylabel("Number of Units")
plt.title("Unit counts by Cell Type")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Spike Raster Plot for Example Trials and Units
# 
# Next, we visualize spike times from a subset of **around 30 units** over **20 trials**, with trial boundaries shown as vertical dashed lines. This provides intuition into temporal spiking across trials and neurons.
# 
# *Note:* This loads only a subset of data for visualization clarity and computational efficiency.

# %%
trial_table = nwbfile.trials
trial_start_times = trial_table['start_time'][:]
trial_stop_times = trial_table['stop_time'][:]

max_trials = 20
trial_idxs = range(min(max_trials, len(trial_table.id)))

max_units = 30
unit_idxs = range(min(max_units, len(units_table.id)))

plt.figure(figsize=(15,8))

for j, unit_idx in enumerate(unit_idxs):
    spike_times = units_table['spike_times'][unit_idx]
    mask = np.zeros_like(spike_times, dtype=bool)
    for trial_i in trial_idxs:
        mask |= (spike_times >= trial_start_times[trial_i]) & (spike_times <= trial_stop_times[trial_i])
    selected_spikes = spike_times[mask]
    plt.scatter(selected_spikes, np.full_like(selected_spikes, j), s=2, color='k')

# Draw trial boundaries
for trial_i in trial_idxs:
    plt.axvline(trial_start_times[trial_i], color='red', linestyle='--', alpha=0.3)
    plt.axvline(trial_stop_times[trial_i], color='red', linestyle='--', alpha=0.3)

plt.xlabel("Time (s)")
plt.ylabel("Units")
plt.title("Spike raster across trials (subset)")
plt.tight_layout()
plt.show()

# %% [markdown]
# # Summary
# - This notebook explored Dandiset 000945, containing multi-electrode neural recordings from awake rats during tFUS stimulation protocols
# - We examined recording metadata including subject info, electrodes, and sorted units
# - We visualized the distribution of unit cell types
# - We generated spike raster plots aligned with stimulation trials
# 
# ---
# 
# **Note:** This notebook serves as a starting point for exploring this dataset. Advanced analyses or scientific conclusions require further domain expertise and statistical validation.