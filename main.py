import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from alzheimer_dataset import AlzheimerDataset
from cnn import CNN
from train import train_model
from test import test_model


if __name__ == '__main__':

    # Definindo transformações
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Carregando os dados
    train_dataset = AlzheimerDataset('./data/train', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_dataset = AlzheimerDataset('./data/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Treinamento do Modelo
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Iniciando o Treinamento
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)

    # Carregando os dados de teste
    test_dataset = AlzheimerDataset('./data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Avaliando o modelo
    test_model(model, test_loader, criterion)