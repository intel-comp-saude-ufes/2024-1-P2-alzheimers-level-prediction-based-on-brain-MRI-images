import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import sklearn.metrics as metrics
from alzheimer_dataset import AlzheimerDataset
from proposed_cnn import ProposedCNN
from train import train_model
from test import test_model
from plots import plot_confusion_matrix, plot_roc_curve



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
    train_data_path = "./data/train"
    # train_data_path = "./train_augmented"
    test_data_path = "./data/test"

    # Defining train transformations
    train_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),                          # when using a model with input for 3 channels
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    # when using a model with input for 3 channels
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    # Loading training data
    train_dataset = AlzheimerDataset(train_data_path, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Chossing the model
    model = ProposedCNN().to(device)
    # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    #Training the model
    train_model(device, model, train_loader, None, criterion, optimizer, num_epochs, early_stopping=True, n_iter_no_change=n_iter_no_change, tol=tol, validate=False, plot_loss_curve=True)
    
    # Saving the trained model
    torch.save(model.state_dict(), 'model.pth')

    print("\nTesting...")

    # Defining transformations
    test_transform = transforms.Compose([
        # transforms.Grayscale(num_output_channels=3),                          # when using a model with input for 3 channels
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])    # when using a model with input for 3 channels
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])


    # Loading test data
    test_dataset = AlzheimerDataset(test_data_path, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    # Test the model
    test_accuracy, test_loss, all_labels, all_preds, all_probs = test_model(device, model, test_loader, criterion)

    print(f'\nTest Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')

    # Plot confusion matrix and ROC curve
    class_names = ['Non', 'Very mild', 'Mild', 'Moderate']
    plot_confusion_matrix(all_labels, all_preds, class_names, save_plot=True)
    plot_roc_curve(all_labels, all_probs, class_names, save_plot=True)

    # Calculate all metrics (Acuracy, Recall, Precision and F1-score)
    results = metrics.classification_report(all_labels, all_preds, target_names=class_names)
    print(results)