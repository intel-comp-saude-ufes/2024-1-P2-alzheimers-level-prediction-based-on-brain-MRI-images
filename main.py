from sklearn.model_selection import StratifiedKFold
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import torch.nn as nn

from alzheimer_dataset import AlzheimerDataset
from simple_cnn import SimpleCNN
from advanced_cnn import AdvancedCNN
from train import train_model, evaluate_model
from test import test_model
from plots import plot_confusion_matrix, plot_roc_curve


# Função para realizar a validação cruzada
def cross_validate_model(device, dataset, model_class, criterion, optimizer_class, num_epochs=25, n_splits=5):
    # Define a validação cruzada com 5 folds
    skf = StratifiedKFold(n_splits=n_splits)
    all_labels = []
    all_preds = []
    fold = 1
    
    # Divide os dados em treinamento e validação para cada fold
    for train_idx, val_idx in skf.split(dataset.image_paths, dataset.labels):
        print(f'\nFold {fold}/{n_splits}')
        
        # Cria os subconjuntos de treinamento e validação
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
        
        # Instancia o modelo e o otimizador
        model = model_class().to(device)
        optimizer = optimizer_class(model.parameters(), lr=0.001)
        
        # Treina o modelo
        train_model(device, model, train_loader, val_loader, criterion, optimizer, num_epochs)

        # Salva o modelo após cada fold
        torch.save(model.state_dict(), f'model_fold_{fold}.pth')
        
        # Avalia o modelo no conjunto de validação
        _, _, labels, preds = evaluate_model(device, model, val_loader, criterion)
        
        # Armazena as etiquetas e previsões de todos os folds
        all_labels.extend(labels)
        all_preds.extend(preds)
        
        fold += 1
    
    return all_labels, all_preds


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


    print("\nTraining model...")

    # Loading training data
    train_dataset = AlzheimerDataset('./data/train', transform=transform)

    # Defining the loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # Realiza a validação cruzada
    all_labels, all_preds = cross_validate_model(
        device, train_dataset, AdvancedCNN, criterion, torch.optim.Adam, num_epochs=2, n_splits=2
    )

    # # Plotting confusion matrix
    # class_names = ['non-demented', 'very-mild-demented', 'mild-demented', 'moderate-demented']
    # plot_confusion_matrix(all_labels, all_preds, class_names)

    # # Plotting ROC curve
    # plot_roc_curve(all_labels, all_preds, class_names)


    print("\nTesting model...")

    # Carrega os dados de teste
    test_dataset = AlzheimerDataset('./data/test', transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Instancia o modelo
    model = AdvancedCNN().to(device)

    # Carrega o modelo treinado
    model.load_state_dict(torch.load('model_fold_1.pth'))

    # Avalia o modelo no conjunto de teste
    labels, preds = test_model(device, model, test_loader, criterion)

    class_names = ['non-demented', 'very-mild-demented', 'mild-demented', 'moderate-demented']

    # Plota a matriz de confusão
    plot_confusion_matrix(labels, preds, class_names)

    # Plota a curva ROC
    plot_roc_curve(labels, preds, class_names)
