"""
Script to explore spike patterns in NWB file
Goals:
1. Show trial-aligned spike rasters for sample units
2. Create PSTHs for different cell types
3. Compare RSU vs FSU response patterns
"""

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()

# Load the NWB file with progress reporting
print("Loading NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file(
    "https://lindi.neurosift.org/dandi/dandisets/000945/assets/a7549e3f-9b14-432a-be65-adb5f6811343/nwb.lindi.json"
)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()
print("NWB file loaded successfully")

# Get basic info
units = nwb.units
trials = nwb.intervals["trials"]
cell_types = units["celltype_label"][:]

# Select subset of trials (first 50) to reduce processing time
trial_starts = trials["start_time"][:50]
trial_ends = trials["stop_time"][:50]

# Select sample units (5 RSU and 5 FSU)
rsu_units = [i for i, ct in enumerate(cell_types) if ct == 1][:5]
fsu_units = [i for i, ct in enumerate(cell_types) if ct == 2][:5]

# Create raster plots
def plot_raster(unit_indices, title):
    plt.figure(figsize=(10, 6))
    for i, unit_idx in enumerate(unit_indices):
        spikes = units["spike_times"][unit_idx]
        for trial_idx, (start, end) in enumerate(zip(trial_starts, trial_ends)):
            trial_spikes = spikes[(spikes >= start) & (spikes <= end)] - start
            plt.scatter(trial_spikes, [trial_idx]*len(trial_spikes), 
                       s=1, color=f'C{i}')
    plt.xlabel("Time from trial start (s)")
    plt.ylabel("Trial number")
    plt.title(title)
    plt.savefig(f"tmp_scripts/{title.lower().replace(' ', '_')}_raster.png")

print("Creating raster plots...")
plot_raster(rsu_units[:3], "RSU Units Raster")
plot_raster(fsu_units[:3], "FSU Units Raster")

# Create PSTHs
def create_psth(unit_indices, cell_type):
    bins = np.linspace(0, trial_ends[0]-trial_starts[0], 20)
    all_counts = []
    
    for unit_idx in unit_indices:
        spikes = units["spike_times"][unit_idx]
        counts = []
        for start, end in zip(trial_starts, trial_ends):
            trial_spikes = spikes[(spikes >= start) & (spikes <= end)] - start
            hist, _ = np.histogram(trial_spikes, bins=bins)
            counts.append(hist)
        all_counts.append(np.mean(counts, axis=0))
    
    plt.figure(figsize=(10, 4))
    for i, counts in enumerate(all_counts):
        plt.plot(bins[:-1], counts, label=f"Unit {unit_indices[i]}")
    plt.xlabel("Time from trial start (s)")
    plt.ylabel("Mean spike count")
    plt.title(f"{cell_type} PSTH")
    plt.legend()
    plt.savefig(f"tmp_scripts/{cell_type.lower()}_psth.png")

print("Creating PSTHs...")
create_psth(rsu_units[:3], "RSU")
create_psth(fsu_units[:3], "FSU")

print("Analysis complete")