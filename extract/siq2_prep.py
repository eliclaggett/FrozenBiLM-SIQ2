import json
import pandas as pd
import pickle
import torch



df = pd.read_csv('/home/eli/code/datasets/SIQ2/train.csv')

video_pths = []
feat_pths = []

for i, row in df.iterrows():
    video_pths.append('/home/eli/code/datasets/SIQ2/video/'+row['video_id'] + '.mp4')
    feat_pths.append('/home/eli/code/datasets/SIQ2/bilm_vis_feats/'+row['video_id'] + '.npy')

feats_df = pd.DataFrame(data={'video_path':video_pths, 'feature_path': feat_pths})

feats_df.to_csv('/home/eli/code/datasets/SIQ2/train_extract_feats.csv')