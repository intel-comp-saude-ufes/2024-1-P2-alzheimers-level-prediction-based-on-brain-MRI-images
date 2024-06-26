import torch
from matplotlib import pyplot as plt

def train_model(device, model, train_loader, val_loader, criterion, optimizer, num_epochs=25, validate=False, plot_loss_curve=False):
    epochs_losses = []

    # Iterate over the specified number of epochs
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        # Initialize epoch training loss
        running_loss = 0.0

        # Iterate over the training data set
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad() # Reset optimizer gradients
            outputs = model(images) # Make prediction using the model
            loss = criterion(outputs, labels) # Calculate batch loss
            loss.backward() # Backpropagation (compute gradients)
            optimizer.step() # Update model parameters
            running_loss += loss.item() * images.size(0) # Accumulate loss
        
        # Calculate epoch loss
        epoch_loss = running_loss / len(train_loader.dataset)
        epochs_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        if validate:
            # Validation
            model.eval()
            val_loss, val_accuracy, _, _, _ = evaluate_model(device, model, val_loader, criterion)
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

    if plot_loss_curve:
        # Plotar as curvas de aprendizado
        plt.plot(epochs_losses, label="CrossEntropyLoss")
        plt.xlabel("Epochs")
        plt.ylabel("Cross Entropy Loss")
        plt.legend()
        plt.show()


def evaluate_model(device, model, data_loader, criterion):
    # Configures the model for evaluation mode
    model.eval()

    # Initializes variables to calculate test loss and accuracy
    val_running_loss = 0.0
    correct = 0
    total = 0

    # Lists to store all true labels and predicted probabilities
    all_labels = []
    all_preds = []
    all_probs = []

    # Disable gradient computation (saves memory and speeds up the process)
    with torch.no_grad():
        # Iterate over the test data set
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Make prediction using the model
            outputs = model(images)

            # Calculate batch loss
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)

            # Get the class predictions (the class with the highest score)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Stores true labels, predictions and probabilities
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

    # Calculates and prints test accuracy
    val_loss = val_running_loss / len(data_loader.dataset)
    accuracy = correct / total

    return val_loss, accuracy, all_labels, all_preds, all_probs
