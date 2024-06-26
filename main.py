import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from alzheimer_dataset import AlzheimerDataset
from simple_cnn import SimpleCNN
from advanced_cnn import AdvancedCNN
from train import train_model
from test import test_model
from plots import plot_confusion_matrix, plot_roc_curve


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
    model = AdvancedCNN().to(device)

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training the model
    train_model(device, model, train_loader, val_loader, criterion, optimizer, num_epochs=1)

    # Saving the trained model
    torch.save(model.state_dict(), 'alzheimer_model.pth')
    print("\nModel saved!")

    # Loading the test data
    test_dataset = AlzheimerDataset('./data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Evaluating the model
    all_labels, all_preds = test_model(device, model, test_loader, criterion)

    # Plotting confusion matrix
    class_names = ['non-demented', 'very-mild-demented', 'mild-demented', 'moderate-demented']
    plot_confusion_matrix(all_labels, all_preds, class_names)

    # Plotting ROC curve
    plot_roc_curve(all_labels, all_preds, class_names)