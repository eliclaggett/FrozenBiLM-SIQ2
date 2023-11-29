# Get 8 samples for each video and store them in one big dictionary keyed on video id
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


extracted = {}
train = pd.read_csv('/home/eli/code/datasets/SIQ2/train.csv')
for vid in tqdm(train['video_id'].unique(), total=len(train['video_id'].unique())):
    if (os.path.isfile(f'/mnt/d/extracted-features/openface_facialfeatures_video/{vid}.csv')):
        
        # Take action units and face_id from all features
        action_units = pd.read_csv(f'/mnt/d/extracted-features/openface_facialfeatures_video/{vid}.csv', skipinitialspace=True)[['face_id', 'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']]
        
        # Save a sample of 10 faces for each video 
        extracted[vid] = action_units.iloc[np.arange(0, len(action_units), len(action_units) / 10)]

torch.save(extracted, '/home/eli/datasets/SIQ2/face.pth')