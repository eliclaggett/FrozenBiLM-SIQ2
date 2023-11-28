import os
import json
import pandas as pd
import pickle
import torch
import webvtt

DATA_DIR = '/home/eli/code/datasets/SIQ2'

directory = os.fsencode(f'{DATA_DIR}/transcript')

subs = {}
for entry in os.scandir(directory):
    if entry.name.endswith(b".vtt") and entry.is_file():
        print(entry.path)
        allCaptions = []
        for caption in webvtt.read(entry.path):
            for text in caption.text.strip().split("\n"):
                if text not in allCaptions:
                    allCaptions.append(text)
    vid_name = entry.name.decode('UTF-8').split('.')[0]
    subs[vid_name] = [{"start": 0, "end":59, "text":" ".join(allCaptions)}]
pickle.dump(subs, open(f"{DATA_DIR}/subtitles.pkl", "wb"))