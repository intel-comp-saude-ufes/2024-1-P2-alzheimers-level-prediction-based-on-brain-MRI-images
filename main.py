import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from alzheimer_dataset import AlzheimerDataset
from cnn import CNN
from train import train_model
from test import test_model


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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Loading validation data
    val_dataset = AlzheimerDataset('./data/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Instantiating and moving the model to the GPU
    model = CNN().to(device)

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    train_model(device, model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

    # Loading the test data
    test_dataset = AlzheimerDataset('./data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluating the model
    test_model(device, model, test_loader, criterion)