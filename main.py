import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import sklearn.metrics as metrics

from alzheimer_dataset import AlzheimerDataset
from simple_cnn import SimpleCNN
from advanced_cnn import AdvancedCNN
from resnet import ResNet, ResidualBlock
from train import train_model
from test import test_model
from plots import plot_confusion_matrix, plot_roc_curve
from aug import get_mri_augmentation_sequence


if __name__ == '__main__':
    
    # Checking if the GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    print("\nTraining model with all train data...")

    # Hiper-parameters
    learning_rate       = 0.001
    batch_size          = 32
    num_epochs          = 20
    n_iter_no_change    = 3
    tol                 = 0.05

    # Defining the data path
    # train_data_path = "./data/train"
    train_data_path = "./data-OAS/train"
    test_data_path = "./data-OAS/test"

    # Defining train transformations
    train_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        # np.array,
        # get_mri_augmentation_sequence().augment_image,        # AUGMENTATION
        # np.copy,
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # when using 3 channels
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # Defining transformations
    test_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # Loading training data
    train_dataset = AlzheimerDataset(train_data_path, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loading training data
    test_dataset = AlzheimerDataset(test_data_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Chossing the model
    # model = SimpleCNN().to(device)
    model = AdvancedCNN().to(device)
    # model = ResNet(ResidualBlock, [2, 2, 2], num_classes=4).to(device)
    # model = models.resnet18().to(device)                                            # sem pesos pre treinados
    # model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)     # com pesos pre treinados

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #Training the model
    train_model(device, model, train_loader, test_loader, criterion, optimizer, num_epochs, early_stopping=True, n_iter_no_change=n_iter_no_change, tol=tol, validate=True, plot_loss_curve=True)
    
    # Saving the trained model
    torch.save(model.state_dict(), 'model.pth')

    print("\nTesting...")
    
    # Test the model
    test_accuracy, test_loss, all_labels, all_preds, all_probs = test_model(device, model, test_loader, criterion)

    print(f'\nTest Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')

    # Plot confusion matrix and ROC curve
    class_names = ['non-demented', 'very-mild-demented', 'mild-demented', 'moderate-demented']
    plot_confusion_matrix(all_labels, all_preds, class_names, save_plot=True)
    plot_roc_curve(all_labels, all_probs, class_names, save_plot=True)

    # Calculate all metrics (Acuracy, Recall, Precision and F1-score)
    results = metrics.classification_report(all_labels, all_preds, target_names=class_names)
    print(results)
