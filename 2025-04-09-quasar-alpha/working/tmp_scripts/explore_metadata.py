"""
Exploratory script to summarize electrodes and neuron cell types in example NWB file.
Saves bar plot of unit counts by cell type as 'tmp_scripts/celltype_counts.png'.
"""

import remfile
import h5py
import pynwb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter

sns.set_theme()

# Load NWB file from URL using recommended approach
url = "https://api.dandiarchive.org/api/assets/f88a9bec-23d6-4444-8b97-8083e45057c9/download/"
file_obj = remfile.File(url)
f = h5py.File(file_obj)
io = pynwb.NWBHDF5IO(file=f)
nwbfile = io.read()

# List electrodes
electrodes_table = nwbfile.electrodes
num_electrodes = len(electrodes_table.id)
print(f"Number of electrodes: {num_electrodes}")

# List units and cell types
units_table = nwbfile.units
num_units = len(units_table.id)
print(f"Number of units: {num_units}")

cell_types = units_table['celltype_label'][:]
cell_types = [ct.decode() if isinstance(ct, bytes) else str(ct) for ct in cell_types]

counts = Counter(cell_types)
print("Counts of units by cell type:", counts)

# Bar plot of cell type counts
plt.figure(figsize=(8,6))
sns.barplot(x=list(counts.keys()), y=list(counts.values()))
plt.xlabel("Cell type label")
plt.ylabel("Number of units")
plt.title("Unit counts by cell type")
plt.tight_layout()
plt.savefig("tmp_scripts/celltype_counts.png")