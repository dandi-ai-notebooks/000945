# %% [markdown]
# # Exploring Dandiset 000945: Neural Spiking Data in the Awake Rat Somatosensory Cortex Responding to Trials of Transcranial Focused Ultrasound Stimulation
#
# This notebook was AI-generated using dandi-notebook-gen and has not been fully verified.
# Use caution when interpreting the code or results.
#
# ## Overview of Dandiset 000945
#
# This Dandiset contains neural spiking data recorded from the somatosensory cortex of awake rats during transcranial focused ultrasound stimulation (tFUS) trials.
# Researchers tested different pulse repetition frequencies (PRFs) of ultrasound stimulation using a 128-element random array ultrasound transducer.
# Chronic electrophysiological recordings were acquired using 32-channel NeuroNexus electrodes.
# Ultrasound stimulation was delivered every 2.5 seconds with a 10% jitter, and each recording has 500 trials.
# The PRFs tested were 30 Hz, 300 Hz, 1500 Hz, 3000 Hz, and 4500 Hz, each with a 200 microsecond pulse duration and a 67 ms ultrasound duration.
# Some recordings were performed under 2% isoflurane anesthesia for comparison.
#
# All 10 subjects were male rats, implanted with their chronic electrode at 6 months of age and then recordings taken first at 8-10 months, and then some repeats taken at 12 months.
# Within each subject's folder are recordings for the different PRFs.
# Most subjects have 5 recordings within, one for each PRF.
# Some subjects have duplicate recordings taken a few months after the original ones.
# A few recordings were not included due to excessive noise in the recordings.
# Files are named in the format SubjectName_PRF_PulseDuration.
# Each file contains spike time data with the cell type labels included for each neurons, as well as time series data for the onset of each trial of ultrasound stimulation.
#
# ## What this notebook will cover
#
# This notebook will demonstrate how to:
# 1. Load the Dandiset using the DANDI API.
# 2. Access and explore the available assets (NWB files).
# 3. Load metadata from a selected NWB file.
# 4. Load and visualize electrophysiology data (spike times and cell type labels).
# 5. Load and visualize trial information.
#
# ## Required Packages
#
# The following packages are required to run this notebook:
# - pynwb
# - h5py
# - remfile
# - matplotlib
# - numpy
# - pandas
# - seaborn
# - dandi
#
# ## Load Dandiset using the DANDI API
#
# %%
from dandi.dandiapi import DandiAPIClient

# Connect to DANDI archive
client = DandiAPIClient()
dandiset = client.get_dandiset("000945")
assets = list(dandiset.get_assets())

print(f"Found {len(assets)} assets in the dataset")
print("\\nFirst 5 assets:")
for asset in assets[:5]:
    print(f"- {asset.path}")

# %% [markdown]
# ## Load Metadata and Data from an NWB File
#
# We will now load the metadata and some example data from one of the NWB files in the Dandiset to demonstrate how to access the data. We will use the first NWB file in the assets list: `sub-BH497/sub-BH497_ses-20240310T143729_ecephys.nwb`.
#
# **Note:** This NWB file is stored remotely, so we will use `remfile` to stream the data.

# %%
import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme()

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

nwb # (NWBFile)
# %% [markdown]
# Show some basic information about the NWB file:
# %%
nwb.session_description # (str) Awake S1 Stimulation by tFUS
# %%
nwb.identifier # (str) BH498_3000_200_anes
# %%
nwb.session_start_time # (datetime) 2024-03-10T14:37:29-04:00
# %%
nwb.timestamps_reference_time # (datetime) 2024-03-10T14:37:29-04:00
# %% [markdown]
# ## Explore Electrode Groups
# %%
nwb.electrode_groups # (LabelledDict)
# %%
nwb.electrode_groups["shank1"] # (ElectrodeGroup)
# %%
nwb.electrode_groups["shank1"].description # (str) electrode group for shank1
# %%
nwb.electrode_groups["shank1"].location # (str) brain area
# %%
nwb.electrode_groups["shank1"].device # (Device)
# %%
nwb.electrode_groups["shank1"].device.description # (str) A1x32-Poly3-10mm-50-177-Z32
# %%
nwb.electrode_groups["shank1"].device.manufacturer # (str) Neuronexus
# %% [markdown]
# ## Explore Devices
# %%
nwb.devices # (LabelledDict)
# %%
nwb.devices["array"] # (Device)
# %%
nwb.devices["array"].description # (str) A1x32-Poly3-10mm-50-177-Z32
# %%
nwb.devices["array"].manufacturer # (str) Neuronexus
# %% [markdown]
# ## Explore Trials
# %%
nwb.intervals # (LabelledDict)
# %%
nwb.intervals["trials"] # (TimeIntervals)
# %% [markdown]
# Convert to a pandas DataFrame with 500 rows and 2 columns
# %%
nwb.trials.to_dataframe() # (DataFrame)
# %% [markdown]
# Show the first few rows of the pandas DataFrame
# %%
nwb.trials.to_dataframe().head() # (DataFrame)
# %%
nwb.trials.description # (str) tFUS stimulation trial onset and offset
# %%
nwb.trials.colnames # (tuple) ['start_time', 'stop_time']
# %% [markdown]
# ## Explore Electrodes
# %%
nwb.electrodes # (DynamicTable)
# %% [markdown]
# Convert to a pandas DataFrame with 32 rows and 8 columns
# %%
nwb.electrodes.to_dataframe() # (DataFrame)
# %% [markdown]
# Show the first few rows of the pandas DataFrame
# %%
nwb.electrodes.to_dataframe().head() # (DataFrame)
# %%
nwb.electrodes.description # (str) all electrodes
# %%
nwb.electrodes.colnames # (tuple) ['x', 'y', 'z', 'imp', 'location', 'filtering', 'group', 'group_name']
# %% [markdown]
# ## Explore Subject
# %%
nwb.subject # (Subject)
# %%
nwb.subject.age # (str) P24W
# %%
nwb.subject.description # (str) HSD:WI rat
# %%
nwb.subject.sex # (str) M
# %%
nwb.subject.species # (str) Rattus norvegicus
# %%
nwb.subject.subject_id # (str) BH497
# %% [markdown]
# ## Explore Units
# %%
nwb.units # (Units)
# %% [markdown]
# Convert to a pandas DataFrame
# %%
nwb.units.to_dataframe()
# %% [markdown]
# Show the first few rows of the pandas DataFrame
# %%
nwb.units.to_dataframe().head()
# %%
nwb.units.description # (str) units table
# %%
nwb.units.colnames # (tuple) ['spike_times', 'celltype_label']
# %%
nwb.units.waveform_unit # (str) volts
# %% [markdown]
# ## Load and Visualize Spike Times
#
# Here, we will load the spike times and cell type labels from the `units` table and plot the spike times for each cell type.
#
# **Note:** We are only loading the first 10 units to avoid loading too much data.

# %%
units_df = nwb.units.to_dataframe().head(10)
spike_times = units_df['spike_times']
celltype_labels = units_df['celltype_label']

# Plot spike times for each cell type
plt.figure(figsize=(10, 6))
for i in range(len(units_df)):
    plt.scatter(spike_times[i], np.ones_like(spike_times[i]) * i, label=celltype_labels[i], s=5)

plt.xlabel('Time (s)')
plt.ylabel('Unit Index')
plt.title('Spike Times for First 10 Units')
plt.legend(loc='upper right')
plt.show()

# %% [markdown]
# ## Load and Visualize Trial Start and Stop Times
#
# Here, we will load the start and stop times for each trial from the `trials` table and plot them.
#
# **Note:** We are only loading the first 10 trials to avoid loading too much data.

# %%
trials_df = nwb.trials.to_dataframe().head(10)
start_times = trials_df['start_time']
stop_times = trials_df['stop_time']

# Plot trial start and stop times
plt.figure(figsize=(10, 6))
plt.vlines(start_times, ymin=0, ymax=1, color='g', label='Start Time')
plt.vlines(stop_times, ymin=0, ymax=1, color='r', label='Stop Time')
plt.xlabel('Time (s)')
plt.ylabel('Trial')
plt.title('Trial Start and Stop Times for First 10 Trials')
plt.legend()
plt.yticks([])
plt.show()

# %% [markdown]
# ## Summary and Future Directions
#
# This notebook demonstrated how to load and explore data from Dandiset 000945, including:
# - Loading the Dandiset using the DANDI API.
# - Accessing and exploring the available assets (NWB files).
# - Loading metadata from a selected NWB file.
# - Loading and visualizing electrophysiology data (spike times and cell type labels).
# - Loading and visualizing trial information.
#
# Possible future directions for analysis include:
# - Performing more in-depth analysis of the electrophysiology data, such as spike sorting and analysis of firing rates.
# - Investigating the relationship between the ultrasound stimulation and the neural activity.
# - Comparing the neural activity across different PRFs and anesthesia conditions.
# - Analyzing the data from other NWB files in the Dandiset.