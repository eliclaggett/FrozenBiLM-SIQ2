import torch
import numpy as np


face = torch.load('/mnt/d/face.pth')
print(face['zzWaf1z9kgk'].shape)

pose = torch.load('/mnt/d/pose.pth')
print(pose['zzWaf1z9kgk'].shape)