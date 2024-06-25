import torch

def train_model(device, model, train_loader, val_loader, criterion, optimizer, num_epochs=25):
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
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        # Validation
        model.eval()

        # Initialize variables to calculate loss and precision
        val_running_loss = 0.0
        correct = 0
        total = 0

        # Disable gradient computation
        with torch.no_grad():
            # Iterate over the validation data set
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                # Make prediction using the model
                outputs = model(images) 

                # Calculate batch loss
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)

                # Get the class predictions
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Calculate and print validation loss and accuracy
        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')
