# %% [markdown]
# # Exploratory Analysis of DANDI Dataset 000945
#
# **Dataset Title:** Neural Spiking Data in the Awake Rat Somatosensory Cortex Responding to Trials of Transcranial Focused Ultrasound Stimulation
#
# **Citation:** Ramachandran, Sandhya et al. (2025) [DANDI Archive Link](https://dandiarchive.org/dandiset/000945/draft)
#
# **Dataset Description:**
# This dataset contains electrophysiological data recorded from awake rats being stimulated transcranially using ultrasound with varying PRFs. Data include:
# * 32-channel recordings
# * Spike-sorted units labeled by cell type (RSU or FSU)
# * Trial onset/offset times for ultrasound stimulation
#
# **Important:**  
# This notebook was **AI-generated via dandi-notebook-gen** and may contain errors. Validate the code and results before relying on them.
#
# ---

# %% [markdown]
# ## Dandiset Metadata
#
# Loaded from DANDI archive:
#
# * **Version:** draft
# * **License:** CC-BY-4.0
# * **Created:** 2024-03-30
# * **Contributors:** Carnegie Mellon University researchers, supported by NIH grants
# * **About:** Chronic in vivo recordings in rats comparing multiple ultrasound pulse repetition frequencies, awake/anesthetized states.
#
# **Note:** Full metadata and citation info are available on DANDI.

# %%
# List all assets in this Dandiset using DANDI API
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
dandiset = client.get_dandiset("000945")
assets = list(dandiset.get_assets())
print("Number of assets in Dandiset:", len(assets))
for a in assets[:5]:
    print(a.path)

# %% [markdown]
# We focus here on a representative session from subject **BH497**, with example file:
# ```
# https://api.dandiarchive.org/api/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/download/
# ```
# which contains awake trial data during ultrasound stimulation.

# %%
# Load this NWB file using LINDI and pynwb
import lindi
import pynwb

nwb_lindi_url = "https://lindi.neurosift.org/dandi/dandisets/000945/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/nwb.lindi.json"
f = lindi.LindiH5pyFile.from_lindi_file(nwb_lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode="r").read()

print(f"Session description: {nwb.session_description}")
print(f"Session start time: {nwb.session_start_time}")

# %% [markdown]
# ## Experimental Metadata
# 
# - Subject ID: `BH497`
# - Species: *Rattus norvegicus*
# - Sex: Male
# - Session: Awake S1 Stimulation by tFUS
# - Number of ultrasound trials: 500
# - Number of electrodes: 32
# - Number of spike-sorted units: 64 (split equally RSU/FSU)
# 
# Trials occur roughly evenly over ~1250 seconds.

# %%
# Access electrodes
electrodes = nwb.electrodes
print("Electrode column names:", electrodes.colnames)
electrode_ids = electrodes["id"].data[:]
print("Electrode IDs:", electrode_ids)

# %%
# Access trial info
trials = nwb.intervals["trials"]
print("Trial start time range:", trials["start_time"][:5], "...")

# %%
# Access units and cell types
units = nwb.units
unit_ids = units["id"].data[:]
cell_types = units["celltype_label"].data[:]
print("Number of units:", len(unit_ids))
print("Number of RSU cells (label=1):", sum(cell_types == 1))
print("Number of FSU cells (label=2):", sum(cell_types == 2))

# %% [markdown]
# ## Distribution of Spike Counts Across Units

# %%
import matplotlib.pyplot as plt
import numpy as np

spike_counts = []
for idx in range(len(unit_ids)):
    spikes = units['spike_times'][idx]
    spike_counts.append(len(spikes))

plt.figure(figsize=(8,6))
plt.hist(spike_counts, bins=20, color='blue', alpha=0.7)
plt.xlabel('Spike Count per Unit')
plt.ylabel('Number of Units')
plt.title('Distribution of Spike Counts Across Units')
plt.tight_layout()
plt.savefig('spike_count_histogram.png')
plt.close()

from IPython.display import Image
Image('spike_count_histogram.png')

# %% [markdown]
# This histogram shows a right-skewed, multimodal distribution of spike counts, with a dominant cluster around 10,000–15,000 spikes and some units showing higher counts.

# %% [markdown]
# ## Mean Firing Rate Per Unit

# %%
mean_firing_rates = []
for idx in range(len(unit_ids)):
    spikes = units['spike_times'][idx]
    duration = (spikes[-1] - spikes[0]) if len(spikes) > 1 else 1
    mean_firing_rates.append(len(spikes) / max(duration, 1e-8))

plt.figure(figsize=(8,6))
plt.bar(unit_ids, mean_firing_rates, color='orange', alpha=0.7)
plt.xlabel('Unit ID')
plt.ylabel('Mean Firing Rate (Hz)')
plt.title('Mean Firing Rate Per Unit')
plt.tight_layout()
plt.savefig('mean_firing_rate_per_unit.png')
plt.close()

Image('mean_firing_rate_per_unit.png')

# %% [markdown]
# This bar plot illustrates firing rate heterogeneity among units, ranging roughly from 5 Hz up to >40 Hz, with a few outliers.

# %% [markdown]
# ## Stimulus Trial Timing

# %%
start_times = trials["start_time"][:]

plt.figure(figsize=(8,6))
plt.hist(start_times, bins=30, color='green', alpha=0.7)
plt.xlabel('Trial Start Time (s)')
plt.ylabel('Number of Trials')
plt.title('Distribution of Trial Start Times')
plt.tight_layout()
plt.savefig('trial_start_times_histogram.png')
plt.close()

Image('trial_start_times_histogram.png')

# %% [markdown]
# This plot shows that stimulus trials were approximately evenly distributed over the recording duration with no significant gaps.

# %% [markdown]
# ## Next Steps
#
# This notebook provides a starting point for exploring dataset 000945, including:
# - Loading NWB data with LINDI+pynwb
# - Extracting metadata, trial info, spike times
# - Performing basic visualizations of spike count distributions and trial structure
#
# **For detailed scientific analyses**, users should consider:
# - Aligning spikes to stimuli (peristimulus histograms)
# - Comparing activity across PRF conditions
# - Statistical tests of firing rate changes
# - Cell type-specific responses
#
# **Important:** No strong scientific claims should be made solely based on these explorations without rigorous analyses and validation.