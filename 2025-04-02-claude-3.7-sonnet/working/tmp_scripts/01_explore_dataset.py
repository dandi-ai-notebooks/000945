'''
This script explores the Dandiset 000945 structure and metadata.
It will:
1. List all the available assets/files
2. Examine subject and session information
3. Look at the structure of a sample NWB file
'''

import pynwb
import lindi
import numpy as np
import matplotlib.pyplot as plt
from dandi.dandiapi import DandiAPIClient

# Set up plotting to save to file instead of displaying
plt.ioff()  # Turn interactive mode off

print("Getting Dandiset information...")
client = DandiAPIClient()
dandiset = client.get_dandiset("000945")
metadata = dandiset.get_metadata()
print(f"Dandiset Name: {metadata.name}")
print(f"Dandiset Description: {metadata.description}")

# Get all assets
print("\nRetrieving assets...")
assets = list(dandiset.get_assets())
print(f"Total assets: {len(assets)}")

# Organize assets by subject and session
assets_by_subject = {}
for asset in assets:
    path = asset.path
    subject_id = path.split('/')[0]
    if subject_id not in assets_by_subject:
        assets_by_subject[subject_id] = []
    assets_by_subject[subject_id].append(asset)

# Print subject summary
print("\nSubject summary:")
for subject_id, subject_assets in assets_by_subject.items():
    print(f"{subject_id}: {len(subject_assets)} files")

# Choose a sample NWB file to examine (using first file from first subject)
sample_asset = assets[0]
sample_path = sample_asset.path
sample_id = sample_asset.identifier
print(f"\nExamining sample file: {sample_path}")

# Create the Lindi file URL for accessing the NWB file
lindi_url = f"https://lindi.neurosift.org/dandi/dandisets/000945/assets/{sample_id}/nwb.lindi.json"

# Open the NWB file
print("\nOpening NWB file...")
f = lindi.LindiH5pyFile.from_lindi_file(lindi_url)
nwb = pynwb.NWBHDF5IO(file=f, mode='r').read()

# Print basic file information
print(f"Session description: {nwb.session_description}")
print(f"NWB identifier: {nwb.identifier}")
print(f"Session start time: {nwb.session_start_time}")
print(f"Subject ID: {nwb.subject.subject_id}")
print(f"Subject species: {nwb.subject.species}")
print(f"Subject sex: {nwb.subject.sex}")
print(f"Subject age: {nwb.subject.age}")

# Check trials information
trials = nwb.intervals["trials"]
print(f"\nNumber of trials: {len(trials)}")

# Check electrode information
electrodes = nwb.electrodes
print(f"\nNumber of electrodes: {len(electrodes)}")
print(f"Electrode columns: {electrodes.colnames}")

# Check units information
units = nwb.units
print(f"\nNumber of units: {len(units)}")
print(f"Unit columns: {units.colnames}")

# Create a plot of the number of files per subject
plt.figure(figsize=(10, 6))
subjects = list(assets_by_subject.keys())
file_counts = [len(assets_by_subject[subject]) for subject in subjects]
plt.bar(subjects, file_counts)
plt.xlabel('Subject ID')
plt.ylabel('Number of Files')
plt.title('Number of Files per Subject')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("tmp_scripts/subject_file_counts.png")
plt.close()

print("\nExploration completed!")