import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import sklearn.metrics as metrics
from proposed_cnn import ProposedCNN
from alzheimer_dataset import AlzheimerDataset
from plots import plot_confusion_matrix, plot_roc_curve


def test_model(device, model, test_loader, criterion):
    """
    Tests the model on a test data set

    INPUT:
        device (torch.device): Device to run the model on
        model (torch.nn.Module): Model to test
        test_loader (torch.utils.data.DataLoader): DataLoader for the test data
        criterion (torch.nn.Module): Loss function

    OUTPUT:
        test_accuracy (float): Test accuracy
        test_loss (float): Test loss
        all_labels (list): List of true labels
        all_preds (list): List of predicted labels
        all_probs (list): List of predicted probabilities
    """

    # Configures the model for evaluation mode
    model.eval()

    # Initializes variables to calculate test loss and accuracy
    test_running_loss = 0.0
    correct = 0
    total = 0

    # Lists to store all true labels and predicted probabilities
    all_labels = []
    all_preds = []
    all_probs = []
    
    # Disable gradient computation (saves memory and speeds up the process)
    with torch.no_grad():
        # Iterate over the test data set
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Make prediction using the model
            outputs = model(images)

            # Calculate batch loss
            loss = criterion(outputs, labels)
            test_running_loss += loss.item() * images.size(0)

            # Get the class predictions (the class with the highest score)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Stores true labels, predictions and probabilities
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
    
    # Calculates and prints test accuracy
    test_loss = test_running_loss / len(test_loader.dataset)
    test_accuracy = correct / total

    return test_accuracy, test_loss, all_labels, all_preds, all_probs


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
    test_dataset = AlzheimerDataset('./data/test', transform=transform)

    # transform dataset in a DataLoader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # load the pre-trained model
    model = ProposedCNN().to(device)
    model.load_state_dict(torch.load("model.pth"))

    print("\nTesting...")

    test_accuracy, test_loss, all_labels, all_preds, all_probs = test_model(device, model, test_loader, criterion)

    print(f'\nTest Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')

    # Plot confusion matrix and ROC curve
    class_names = ['Non', 'Very mild', 'Mild', 'Moderate']
    plot_confusion_matrix(all_labels, all_preds, class_names, save_plot=True)
    plot_roc_curve(all_labels, all_probs, class_names, save_plot=True)

    # Calculate all metrics (Acuracy, Recall, Precision and F1-score)
    results = metrics.classification_report(all_labels, all_preds, target_names=class_names)
    print(results)