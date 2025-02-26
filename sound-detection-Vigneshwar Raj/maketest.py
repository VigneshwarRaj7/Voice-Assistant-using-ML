import os
import numpy as np
import pandas as pd

from feature_extraction import get_mfcc_features
# Paths to the folders containing audio files for each class
folder_paths = ['test']

num_mfccs = 13
# Assuming a maximum of 'n' frames; you need to determine this from your data
max_frames = 200  # Example value; you need to calculate based on your data

data_rows = []

# Iterate over each folder and process each audio file
for label, folder_path in enumerate(folder_paths):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(folder_path, file_name)
            mfccs = get_mfcc_features(file_path)
            if len(mfccs) != 117: continue
            data_rows.append(np.append(mfccs, label))

df = pd.DataFrame(data_rows, columns=[f"mfcc_{i}" for i in range(len(mfccs))] + ['label'])

# Save the DataFrame to a CSV file
df.to_csv('test.csv', index=False)

