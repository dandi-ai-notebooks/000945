# Plot electrode positions for Dandiset 000945 NWB file

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

url = "https://api.dandiarchive.org/api/assets/a4e04662-e4cb-49f3-9076-41e04e833a11/download/"

file = remfile.File(url)
f = h5py.File(file)
io = pynwb.NWBHDF5IO(file=f)
nwb = io.read()

electrodes = nwb.electrodes

x = electrodes['x'][:]
y = electrodes['y'][:]
z = electrodes['z'][:]

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.set_xlabel('X (um)')
ax.set_ylabel('Y (um)')
ax.set_zlabel('Z (um)')
ax.set_title('Electrode locations (3D)')

plt.tight_layout()
plt.savefig('tmp_scripts/electrode_positions.png')

io.close()