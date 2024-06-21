import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class AlzheimerDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Mapear as pastas para os labels
        self.label_map = {
            'non-demented': 0,
            'mild-demented': 1,
            'moderate-demented': 2,
            'very-mild-demented': 3
        }

        # Coletar todos os caminhos de imagem e seus respectivos labels
        for label_name, label_idx in self.label_map.items():
            label_dir = os.path.join(data_dir, label_name)
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                if os.path.isfile(img_path):  # Verifica se é um arquivo
                    self.image_paths.append(img_path)
                    self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label