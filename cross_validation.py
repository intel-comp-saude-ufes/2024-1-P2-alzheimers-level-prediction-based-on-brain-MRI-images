import numpy as np
from sklearn.model_selection import StratifiedKFold
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn

from alzheimer_dataset import AlzheimerDataset
from simple_cnn import SimpleCNN
from advanced_cnn import AdvancedCNN
from train import train_model
from test import test_model
from plots import plot_confusion_matrix, plot_roc_curve

def cross_validate_model(device, dataset, model_class, criterion, optimizer_class, num_epochs=25, n_splits=5):
    # Sets cross validation with n_splits folds
    skf = StratifiedKFold(n_splits=n_splits)
    all_labels = []
    all_preds = []
    all_probs = []
    accuracys = []
    fold = 1
    
    # Splits data into training and validation for each fold
    for train_idx, val_idx in skf.split(dataset.image_paths, dataset.labels):
        print(f'\nFold {fold}/{n_splits}')
        
        # Creates the training and validation subsets
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # Creates the trainig and validation loaders
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        # Instantiates the model and optimizer
        model = model_class().to(device)
        optimizer = optimizer_class(model.parameters(), lr=0.001)
        
        # Train the model
        train_model(device, model, train_loader, val_loader, criterion, optimizer, num_epochs, early_stopping=True, n_iter_no_change=5, tol=0.01, validate=True, plot_loss_curve=False)
        
        # Evaluate the model on the validation set again to get the confusion matrix and ROC curve
        accuracy, loss, labels, preds, probs = test_model(device, model, val_loader, criterion)
        accuracys.append(accuracy)
        
        # Stores labels, predictions and probs for all folds
        all_labels.extend(labels)
        all_preds.extend(preds)
        all_probs.extend(probs)
        
        fold += 1
    
    return all_labels, all_preds, all_probs, accuracys

if __name__ == '__main__':
    
    # Checking if the GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}\n')

    print("\nCross validation...")

    # Defining transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Loading training data
    train_dataset = AlzheimerDataset('./data_augmented', transform=transform)

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # Optimizer class
    optimizer = torch.optim.Adam
    # Model class
    model = AdvancedCNN
    # Number of epochs
    num_epochs = 20
    # NUmber of folders
    num_splits = 5

    # Perform cross validation
    all_labels, all_preds, all_probs, accuracys = cross_validate_model(
        device, train_dataset, model, criterion, optimizer, num_epochs=num_epochs, n_splits=num_splits
    )

    # Convertendo a lista para um array NumPy
    accuracys_array = np.array(accuracys)
    # Calculando a média
    mean = np.mean(accuracys_array)
    # Calculando o desvio padrão
    std_dev = np.std(accuracys_array)

    print(f"Acurácia: {(mean*100):.2f} % +/- {(std_dev*100):.2f} %")

    # Plot confusion matrix and ROC curve for all folds
    class_names = ['non-demented', 'very-mild-demented', 'mild-demented', 'moderate-demented']
    plot_confusion_matrix(all_labels, all_preds, class_names)
    plot_roc_curve(all_labels, all_probs, class_names)