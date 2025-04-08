# %% [markdown]
# # Neural Spiking Data in the Awake Rat Somatosensory Cortex Responding to Trials of Transcranial Focused Ultrasound Stimulation
#
# This notebook was AI-generated using dandi-notebook-gen and has not been fully verified.
# Please be cautious when interpreting the code or results.
#
# This notebook provides an introduction to the Dandiset 000945, which contains neural spiking data in the awake rat somatosensory cortex responding to trials of transcranial focused ultrasound stimulation.
# The dataset includes recordings from 10 male rats implanted with chronic electrodes in the somatosensory cortex.
# Ultrasound stimulation was delivered at different pulse repetition frequencies (PRFs), and electrophysiological recordings were acquired using 32-channel NeuroNexus electrodes.
#
# The notebook demonstrates how to:
#
# 1.  Load and explore the dataset's structure
# 2.  Access and visualize sample data from NWB files
# 3.  Perform common analyses relevant to the dataset's content
#
# Before using this notebook, please ensure you have installed the necessary packages:
#
# ```bash
# pip install lindi pynwb matplotlib numpy seaborn dandi
# ```

# %%
from dandi.dandiapi import DandiAPIClient
client = DandiAPIClient()
dandiset = client.get_dandiset("000945")
assets = list(dandiset.get_assets())
assets

# %% [markdown]
# The above code block uses the DANDI API to list all of the assets in the Dandiset.

# %% [markdown]
# ## Loading and Exploring the Dataset
#
# The dataset is stored in the Neurodata Without Borders (NWB) format.
# We will use the `pynwb` and `lindi` libraries to load and explore the data. We will be loading remote files which may
# take time.

# %%
import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load the NWB file
try:
    f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000945/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/nwb.lindi.json")
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
    print("NWB file loaded successfully.")
except Exception as e:
    print(f"Error loading NWB file: {e}")

# %% [markdown]
# ## Exploring the NWB File Content
#
# Now we can explore the content of the NWB file.

# %%
nwb.session_description

# %%
nwb.identifier

# %%
nwb.session_start_time

# %% [markdown]
# Getting information about the electrodes and units.

# %%
electrodes = nwb.electrodes
electrodes.colnames

# %%
electrode_ids = electrodes["id"].data[:]
len(electrode_ids)

# %%
units = nwb.units
units.colnames

# %%
unit_ids = units["id"].data
len(unit_ids)

# %% [markdown]
# ## Accessing and Visualizing Sample Data
#
# The following code shows how to access and visualize sample data from the NWB file.
# We will start by plotting the spike times for a subset of units.

# %%
# Get spike times for a subset of units
try:
    units = nwb.units
    unit_ids = units["id"].data[:]
    num_units = len(unit_ids)
    num_units_to_plot = min(5, num_units) # Plot max 5 units
    spike_times = []
    for i in range(num_units_to_plot):
        spike_times.append(units["spike_times"][i][:])

    # Create a raster plot
    plt.figure(figsize=(10, 6))
    for i, spikes in enumerate(spike_times):
        plt.vlines(spikes, i + 0.5, i + 1.5, linewidth=0.5)

    plt.xlabel("Time (s)")
    plt.ylabel("Unit ID")
    plt.yticks(np.arange(1, num_units_to_plot + 1), unit_ids[:num_units_to_plot])
    plt.title("Raster Plot of Spike Times for a Subset of Units")
    plt.savefig("raster_plot.png")

except Exception as e:
    print(f"Error generating raster plot: {e}")

# %% [markdown]
# The raster plot shows the spike times for a subset of units. The x-axis represents time (in seconds), and the y-axis represents the unit ID. Each vertical line represents a spike. The raster plot can be used to visualize the spiking activity of individual units and to identify patterns in the spiking activity.

# %% [markdown]
# The distribution of trial start times can be plotted using the following code. The trials data is useful since ultrasound stimulation is delivered every 2.5 seconds with a 10% jitter, and each recording has 500 trials.

# %%
try:
    # Plot trial start times
    trials = nwb.intervals["trials"]
    trial_start_times = trials["start_time"][:]
    plt.figure(figsize=(10, 4))
    plt.hist(trial_start_times, bins=50)
    plt.xlabel("Trial Start Time (s)")
    plt.ylabel("Count")
    plt.title("Distribution of Trial Start Times")
    plt.savefig("trial_start_times.png")

except Exception as e:
    print(f"Error generating trial start times histogram: {e}")