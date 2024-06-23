import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedCNN(nn.Module):
    def __init__(self):
        super(AdvancedCNN, self).__init__()
        
        # Primeira camada convolucional
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Normalização em lote
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        # Segunda camada convolucional
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Normalização em lote
        
        # Terceira camada convolucional
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Normalização em lote
        
        # Quarta camada convolucional
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)  # Normalização em lote
        
        # Camadas totalmente conectadas
        self.fc1 = nn.Linear(256 * 14 * 14, 512)  # Ajuste as dimensões conforme necessário
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 4)  # 4 classes: non-demented, very mild demented, mild demented, demented
    
    def forward(self, x):
        # Passa pela primeira camada convolucional, normalização, ativação e pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Passa pela segunda camada convolucional, normalização, ativação e pooling
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Passa pela terceira camada convolucional, normalização, ativação e pooling
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Passa pela quarta camada convolucional, normalização, ativação e pooling
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Achatar (flatten) as ativações para a camada totalmente conectada
        x = x.view(-1, 256 * 14 * 14)
        
        # Passa pelas camadas totalmente conectadas
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x