# This script loads the example NWB file using LINDI and pynwb,
# and generates exploratory plots:
# 1. Histogram of total spike counts across all units.
# 2. Mean firing rate per unit.
# 3. Distribution of stimulus trial start times.
# The plots are saved in tmp_scripts/ for later inclusion in the notebook.

import lindi
import pynwb
import numpy as np
import matplotlib.pyplot as plt

nwb_lindi_url = "https://lindi.neurosift.org/dandi/dandisets/000945/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/nwb.lindi.json"

f = lindi.LindiH5pyFile.from_lindi_file(nwb_lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode="r").read()

units = nwb.units
unit_ids = units['id'].data[:]

spike_counts = []
mean_firing_rates = []

for idx in range(len(unit_ids)):
    spikes = units['spike_times'][idx]
    spike_counts.append(len(spikes))
    duration = nwb.session_start_time.timestamp()  # session_start_time is timezone-aware datetime
    if 'timestamps_reference_time' in nwb.fields:
        duration = (spikes[-1] - spikes[0]) if len(spikes) > 1 else 1
    mean_firing_rates.append(len(spikes) / max(duration, 1e-8))

# Plot histogram of spike counts across units
plt.figure(figsize=(8, 6))
plt.hist(spike_counts, bins=20, color='blue', alpha=0.7)
plt.xlabel('Spike Count per Unit')
plt.ylabel('Number of Units')
plt.title('Distribution of Spike Counts Across Units')
plt.tight_layout()
plt.savefig('tmp_scripts/spike_count_histogram.png')
plt.close()

# Plot mean firing rates per unit
plt.figure(figsize=(8, 6))
plt.bar(unit_ids, mean_firing_rates, color='orange', alpha=0.7)
plt.xlabel('Unit ID')
plt.ylabel('Mean Firing Rate (Hz)')
plt.title('Mean Firing Rate Per Unit')
plt.tight_layout()
plt.savefig('tmp_scripts/mean_firing_rate_per_unit.png')
plt.close()

# Plot distribution of trial start times
trials = nwb.intervals["trials"]
start_times = trials["start_time"][:]

plt.figure(figsize=(8, 6))
plt.hist(start_times, bins=30, color='green', alpha=0.7)
plt.xlabel('Trial Start Time (s)')
plt.ylabel('Number of Trials')
plt.title('Distribution of Trial Start Times')
plt.tight_layout()
plt.savefig('tmp_scripts/trial_start_times_histogram.png')
plt.close()