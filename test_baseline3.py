from diffusion import Diffusion
from dataloader import AnitaDataset
import torch

# Quick test
model = Diffusion(device='cuda')
print("Model loaded!")

dataset = AnitaDataset(root_dir="./data/data_split/train", between_frames=3)
print(f"Dataset size: {len(dataset)}")

sample = dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Start shape: {sample['anchor_start'].shape}")
print(f"Targets shape: {sample['targets'].shape}")

print("Everything looks good!")