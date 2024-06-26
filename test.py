import torch

def test_model(device, model, test_loader, criterion):
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
    print(f'\nTest Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')

    return all_labels, all_preds, all_probs
