import os
from torch.utils.data import Dataset
from PIL import Image

class AlzheimerDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Map folders to labels
        self.label_map = {
            'non-demented': 0,
            'very-mild-demented': 1,
            'mild-demented': 2,
            'moderate-demented': 3
        }

        # Collect all image paths and their respective labels
        for label_name, label_idx in self.label_map.items():
            label_dir = os.path.join(data_dir, label_name)
            for img_name in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_name)
                if os.path.isfile(img_path): # Check if it is a file
                    self.image_paths.append(img_path)
                    self.labels.append(label_idx)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L') # Convert to grayscale
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label