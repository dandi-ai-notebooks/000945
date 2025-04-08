# This script loads an example NWB file from the Dandiset using LINDI and pynwb.
# It extracts and prints basic info: metadata, number of trials, electrodes, units,
# and a summary of the cell types in the recording.

import lindi
import pynwb
import numpy as np

nwb_lindi_url = "https://lindi.neurosift.org/dandi/dandisets/000945/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/nwb.lindi.json"

f = lindi.LindiH5pyFile.from_lindi_file(nwb_lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode="r").read()

print(f"Session description: {nwb.session_description}")
print(f"Identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")

trials = nwb.intervals["trials"]
print(f"Number of trials: {len(trials['id'])}")

electrodes = nwb.electrodes
electrode_ids = electrodes['id'].data[:]
print(f"Number of electrodes: {len(electrode_ids)}")
print(f"Electrode IDs: {electrode_ids}")

units = nwb.units
unit_ids = units['id'].data[:]
cell_types = units['celltype_label'].data[:]

print(f"Number of units: {len(unit_ids)}")
print(f"Unit IDs: {unit_ids}")

num_rsu = np.sum(cell_types == 1)
num_fsu = np.sum(cell_types == 2)
print(f"Number of RSU cells (label=1): {num_rsu}")
print(f"Number of FSU cells (label=2): {num_fsu}")

# Summarize spike times shape for first 5 units
for idx in range(min(5, len(unit_ids))):
    spikes = units['spike_times'][idx]
    print(f"Unit index {idx}, ID {unit_ids[idx]} spike count: {len(spikes)}")