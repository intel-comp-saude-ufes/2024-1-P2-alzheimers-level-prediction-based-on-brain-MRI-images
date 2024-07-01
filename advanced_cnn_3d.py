import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedCNN3D(nn.Module):
    def __init__(self, num_classes=4):
        super(AdvancedCNN3D, self).__init__()
        self.N = 61
        self.W = 496
        self.H = 128

        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 7 * 62 * 16, 256)  # Ajuste necessário aqui
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # print(f'Input shape: {x.shape}')  # Debug: print input shape
        x = self.pool(F.relu(self.conv1(x)))
        # print(f'After conv1 and pool: {x.shape}')  # Debug: print shape after conv1 and pool
        x = self.pool(F.relu(self.conv2(x)))
        # print(f'After conv2 and pool: {x.shape}')  # Debug: print shape after conv2 and pool
        x = self.pool(F.relu(self.conv3(x)))
        # print(f'After conv3 and pool: {x.shape}')  # Debug: print shape after conv3 and pool

        x = x.view(x.size(0), -1)  # Flatten the tensor
        # print(f'After flatten: {x.shape}')  # Debug: print shape after flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Testando o modelo
N, W, H = 61, 496, 128  # Dimensões do seu volume de entrada
model = AdvancedCNN3D()
sample_input = torch.randn(1, 1, N, W, H)  # Exemplo de entrada
model(sample_input)