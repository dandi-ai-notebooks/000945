"""
This script explores the trial structure in an NWB file from Dandiset 000945.
It visualizes the timing of trials and the intervals between trials.
"""

import pynwb
import h5py
import remfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Load the NWB file
url = "https://api.dandiarchive.org/api/assets/b88188c8-4e4c-494c-8dab-806b1efd55eb/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

# Get trials dataframe
trials_df = nwb.trials.to_dataframe()

# Calculate trial durations and inter-trial intervals
trial_durations = trials_df['stop_time'] - trials_df['start_time']
inter_trial_intervals = trials_df['start_time'].iloc[1:].reset_index(drop=True) - trials_df['stop_time'].iloc[:-1].reset_index(drop=True)

print(f"Number of trials: {len(trials_df)}")
print(f"Average trial duration: {np.mean(trial_durations):.6f} seconds")
print(f"Average inter-trial interval: {np.mean(inter_trial_intervals):.6f} seconds")
print(f"Min inter-trial interval: {np.min(inter_trial_intervals):.6f} seconds")
print(f"Max inter-trial interval: {np.max(inter_trial_intervals):.6f} seconds")

# Plot the trial start times throughout the recording
plt.figure(figsize=(10, 4))
plt.plot(trials_df.index, trials_df['start_time'], 'o-', markersize=3)
plt.xlabel('Trial Number')
plt.ylabel('Start Time (s)')
plt.title('Trial Start Times Throughout Recording')
plt.grid(True)
plt.savefig('explore/trial_start_times.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot histogram of inter-trial intervals
plt.figure(figsize=(10, 4))
plt.hist(inter_trial_intervals, bins=30, alpha=0.7)
plt.xlabel('Inter-trial Interval (s)')
plt.ylabel('Count')
plt.title('Distribution of Inter-trial Intervals')
plt.grid(True)
plt.savefig('explore/inter_trial_intervals_hist.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot the first 10 trials as a timeline
plt.figure(figsize=(12, 3))
for i in range(10):
    plt.plot([trials_df['start_time'].iloc[i], trials_df['stop_time'].iloc[i]], [i, i], 'b-', linewidth=2)
    if i < 9:  # Add inter-trial gap visualization
        plt.plot([trials_df['stop_time'].iloc[i], trials_df['start_time'].iloc[i+1]], [i, i+1], 'r--', alpha=0.5)
plt.yticks(range(10), [f'Trial {i+1}' for i in range(10)])
plt.xlabel('Time (s)')
plt.title('Timeline of First 10 Trials')
plt.grid(True, alpha=0.3)
plt.savefig('explore/first_10_trials_timeline.png', dpi=300, bbox_inches='tight')
plt.close()

io.close()
h5_file.close()
remote_file.close()

print("Plots saved to explore directory.")