import os
from glob import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
from PIL import Image

class AlzheimerDataset3D(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.classes = sorted(os.listdir(data_path))
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(data_path, class_name)
            patient_dirs = sorted(os.listdir(class_path))
            for patient_dir in patient_dirs:
                patient_path = os.path.join(class_path, patient_dir)
                slices = sorted(glob(os.path.join(patient_path, '*.jpg')))
                # print(f"Class: {class_name}, Patient: {patient_dir}, Number of slices: {len(slices)}")  # Verificação
                if slices:  # Ensure there are slices
                    self.samples.append((slices, class_idx))
                
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        slices, class_idx = self.samples[idx]
        volume = []
        for slice_path in slices:
            img = Image.open(slice_path).convert('L')  # Convert to grayscale
            if self.transform:
                img = self.transform(img)
            volume.append(img)
        if not volume:  # Ensure volume is not empty
            raise ValueError(f"No images found for index {idx}")
        volume = torch.stack(volume)  # Convert to tensor
        volume = volume.permute(1, 0, 2, 3)  # Rearrange to (channels, depth, height, width)
        return volume, class_idx
