import pynwb
import lindi
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# Load the NWB file
try:
    f = lindi.LindiH5pyFile.from_lindi_file("https://lindi.neurosift.org/dandi/dandisets/000945/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/nwb.lindi.json")
    nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
except Exception as e:
    print(f"Error loading NWB file: {e}")
    exit()

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
    plt.savefig("tmp_scripts/raster_plot.png")
    plt.close()

    print("Raster plot generated successfully at tmp_scripts/raster_plot.png")

except Exception as e:
    print(f"Error generating raster plot: {e}")

try:
    # Plot trial start times
    trials = nwb.intervals["trials"]
    trial_start_times = trials["start_time"][:]
    plt.figure(figsize=(10, 4))
    plt.hist(trial_start_times, bins=50)
    plt.xlabel("Trial Start Time (s)")
    plt.ylabel("Count")
    plt.title("Distribution of Trial Start Times")
    plt.savefig("tmp_scripts/trial_start_times.png")
    plt.close()

    print("Trial start times histogram generated successfully at tmp_scripts/trial_start_times.png")

except Exception as e:
    print(f"Error generating trial start times histogram: {e}")