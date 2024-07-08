import torch
from matplotlib import pyplot as plt
import numpy as np

from test import test_model

# from torch.utils.tensorboard import SummaryWriter

def train_model(device, model, train_loader, val_loader, criterion, optimizer, num_epochs=25, early_stopping=False, n_iter_no_change=3, tol=0.1, validate=False, plot_loss_curve=False):
    epochs_losses_train     = []
    epochs_losses_val       = []
    epochs_accuracies_train = []
    epochs_accuracies_val   = []

    n_iter_no_change_actual = 0
    best_loss = np.inf

    # Iterate over the specified number of epochs
    for epoch in range(num_epochs):

        # Set the model to training mode
        model.train()

        # Initialize epoch training loss and correct predictions
        running_loss    = 0.0
        correct_train   = 0
        total_train     = 0

        # Iterate over the training data set
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)             # Make prediction using the model
            loss = criterion(outputs, labels)   # Calculate batch loss
            loss.backward()                     # Backpropagation (compute gradients)
            optimizer.step()                    # Update model parameters

            running_loss += loss.item()         # Accumulate loss

            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        # Calculate epoch loss and accuracy
        epoch_loss_train = running_loss / len(train_loader)
        epochs_losses_train.append(epoch_loss_train)

        epoch_accuracy_train = 100 * correct_train / total_train
        epochs_accuracies_train.append(epoch_accuracy_train)

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss_train:.4f}, Accuracy: {epoch_accuracy_train:.2f}%')

        # Validation
        if validate:
            model.eval()
            val_accuracy, val_loss, _, _, _ = test_model(device, model, val_loader, criterion)
            epochs_losses_val.append(val_loss)
            epochs_accuracies_val.append(val_accuracy * 100)
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy*100:.2f}%')

        # Saving the checkpoint of the last epoch
        checkpoint = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':                 criterion,
            'loss_val':             epoch_loss_train
        }        
        torch.save(checkpoint, "last_checkpoint.pth")

        # Saving the best result (lowest loss)
        if epoch_loss_train < best_loss:
            best_loss = loss
            torch.save(checkpoint, "best_checkpoint.pth")

        # Check if early stopping should be activated
        if early_stopping and epoch >= 1:
            if ( (epochs_losses_train[epoch-1] - epochs_losses_train[epoch]) < tol ):
                n_iter_no_change_actual += 1
            else:
                n_iter_no_change_actual = 0

            if n_iter_no_change_actual == n_iter_no_change:
                print(f'Early stopping activated at epoch {epoch+1}!')
                break

    # Plotting the accuracy and loss curves at the finish of the training
    if plot_loss_curve:
        plt.figure(figsize=(14, 5))

        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs_accuracies_train, label="train")
        # plt.plot(epochs_accuracies_val, label="test")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("Model Accuracy")

        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs_losses_train, label="train")
        # plt.plot(epochs_losses_val, label="test")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Model Loss")

        plt.tight_layout()
        plt.savefig("accuracy_loss_curves.png")
