import torch
from matplotlib import pyplot as plt
import numpy as np

from test import test_model

# from torch.utils.tensorboard import SummaryWriter

def train_model(device, model, train_loader, val_loader, criterion, optimizer, num_epochs=25, early_stopping=False, n_iter_no_change=3, tol=0.1, validate=False, plot_loss_curve=False):
    epochs_losses = []
    n_iter_no_change_actual = 0
    best_loss = np.inf

    # writer = SummaryWriter("runs/alz")

    # Iterate over the specified number of epochs
    for epoch in range(num_epochs):
        # Set the model to training mode
        model.train()

        # Initialize epoch training loss
        running_loss = 0.0

        # Iterate over the training data set
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images) # Make prediction using the model
            loss = criterion(outputs, labels) # Calculate batch loss
            
            # Backward and optimize
            optimizer.zero_grad() # Reset optimizer gradients
            loss.backward() # Backpropagation (compute gradients)
            optimizer.step() # Update model parameters

            running_loss += loss.item() * images.size(0) # Accumulate loss
        
        # Calculate epoch loss
        epoch_loss = running_loss / len(train_loader.dataset)
        epochs_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Check if early stopping should be activated
        if early_stopping and epoch >= 1:
            if ( (epochs_losses[epoch-1] - epochs_losses[epoch]) < tol ):
                n_iter_no_change_actual += 1
            else:
                n_iter_no_change_actual = 0

            if n_iter_no_change_actual == n_iter_no_change:
                print(f'Early stopping activated at epoch {epoch+1}!')
                break
        
        # Validation
        if validate:
            model.eval()
            val_accuracy, val_loss, _, _, _ = test_model(device, model, val_loader, criterion)
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
            
        #     temp = {
        #         "Training loss": epoch_loss,
        #         "Validation loss": val_loss
        #     }
        #     writer.add_scalars("Loss", temp, epoch)

        # else:
        #     writer.add_scalar("Training loss", epoch_loss, epoch)
            

        # Saving the checkpoint of the last epoch
        checkpoint = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss':                 criterion,
            'loss_val':             epoch_loss
        }        
        torch.save(checkpoint, "last_checkpoint.pth")

        # Saving the best result (lowest loss)
        if epoch_loss < best_loss:
            best_loss = loss
            torch.save(checkpoint, "best_checkpoint.pth")

    # Ploting the loss curve at the finish of the train
    if plot_loss_curve:
        plt.plot(epochs_losses, label="CrossEntropyLoss")
        plt.xlabel("Epochs")
        plt.ylabel("Cross Entropy Loss")
        plt.legend()
        plt.savefig("cross_entropy_loss.png")

    # Flush all info to the TensorBoard
    # writer.close()
