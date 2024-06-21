import torch

def test_model(device, model, test_loader, criterion):
    model.eval()
    test_running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_loss = test_running_loss / len(test_loader.dataset)
    test_accuracy = correct / total
    print(f'\nTest Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
