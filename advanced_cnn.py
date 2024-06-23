import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch Normalization
        
        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)  # Batch Normalization
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Adjust dimensions as needed
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)  # 4 classes: non-demented, very mild demented, mild demented, demented
    

    def forward(self, x):
        # Goes through layer 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Goes through layer 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Goes through layer 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Goes through layer 4
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten activations for the fully connected layer
        x = x.view(-1, 256 * 14 * 14)
        
        # Goes through fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x