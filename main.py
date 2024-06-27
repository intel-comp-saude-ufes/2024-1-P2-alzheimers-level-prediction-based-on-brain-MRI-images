import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from alzheimer_dataset import AlzheimerDataset
from simple_cnn import SimpleCNN
from advanced_cnn import AdvancedCNN
from resnet import ResNet, ResidualBlock
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

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    print("\nTraining model with all data...")
    
    # Genereating final model

    model = AdvancedCNN().to(device)

    #model = ResNet(ResidualBlock(in_channels=1), [2, 2, 2], num_classes=4)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_epochs = 10

    train_model(device, model, train_loader, None, criterion, optimizer, num_epochs, validate=False, plot_loss_curve=True)
    torch.save(model.state_dict(), 'model.pth') # Save the final model

    print("\nTesting...")

    # Loading training data
    test_dataset = AlzheimerDataset('./data/test', transform=transform)

    # transform dataset in a DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    test_accuracy, test_loss, all_labels, all_preds, all_probs = test_model(device, model, test_loader, criterion)

    print(f'\nTest Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')

    # Plot confusion matrix and ROC curve
    class_names = ['non-demented', 'very-mild-demented', 'mild-demented', 'moderate-demented']
    plot_confusion_matrix(all_labels, all_preds, class_names)
    plot_roc_curve(all_labels, all_probs, class_names)