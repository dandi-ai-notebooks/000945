import pynwb
import h5py
import remfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme()

# This script explores the trials data in the NWB file.

url = "https://api.dandiarchive.org/api/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/download/"
remote_file = remfile.File(url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file, mode="r")
nwb = io.read()

# Get the trials data
trials = nwb.trials
start_time = trials.start_time[:]
stop_time = trials.stop_time[:]

# Calculate the duration of each trial
trial_duration = stop_time - start_time

# Plot the trial durations
plt.figure(figsize=(10, 5))
plt.hist(trial_duration, bins=[2.1, 2.2, 2.3])
plt.xlabel("Trial Duration (s)")
plt.ylabel("Number of Trials")
plt.title("Distribution of Trial Durations")
plt.xlim(2.0, 2.4)  # Set x-axis limits for better visualization.
plt.savefig("explore/trial_durations.png")

print("Trial durations:", trial_duration)
print("Mean trial duration:", np.mean(trial_duration))