# Get 8 samples for each video and store them in one big dictionary keyed on video id
import os
import glob
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def ravel_index(x, dims):
    i = 0
    for dim, j in zip(dims, x):
        i *= dim
        i += j
    return i

extracted = {}
train = pd.read_csv('/home/eli/code/datasets/SIQ2/train.csv')
for vid in tqdm(train['video_id'].unique(), total=len(train['video_id'].unique())):
    pose_dir = f'/mnt/d/extracted-features/openpose_skeletons/{vid}'
    if (os.path.isdir(pose_dir)):
        f_list = glob.glob(f'{pose_dir}/*.npy')
        f_list = [f[f.rindex('/'):] for f in f_list]
        frames = [int(f[f.rindex('_')+1:f.rindex('.')]) for f in f_list]
        sample = torch.zeros((10,76))
        if (len(frames) > 0):
            max_frame = np.max(frames)
            step = 0
            for i in np.arange(0, max_frame, max_frame / 5):
                idx = int(i)
                pose = np.load(f'{pose_dir}/{vid}_{idx}.npy')
                if (pose.shape[0] > 0):
                    pose = pose[0:2,:,:] # Only use two bodies
                    body1 = np.concatenate(([0],pose[0].flatten()))
                    sample[0+step] = torch.as_tensor(body1)
                    if (len(pose) > 1):
                        body2 = np.concatenate(([1],pose[1].flatten()))
                        sample[1+step] = torch.as_tensor(body2)
                step += 2
        
        extracted[vid] = sample

torch.save(extracted, '/mnt/d/pose.pth')