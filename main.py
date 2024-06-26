import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from alzheimer_dataset import AlzheimerDataset
from simple_cnn import SimpleCNN
from advanced_cnn import AdvancedCNN
from train import train_model


if __name__ == '__main__':
    
    # Checking if the GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    # Defining transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Loading training data
    train_dataset = AlzheimerDataset('./data/train', transform=transform)

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    print("\nTraining model with all data...")
    
    # Genereating final model
    model = AdvancedCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    num_epochs = 5
    train_model(device, model, train_loader, None, criterion, optimizer, num_epochs, validate=False, plot_loss_curve=True)
    torch.save(model.state_dict(), 'model.pth') # Save the final model