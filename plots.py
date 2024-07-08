import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, class_names, save_plot=False):
    # Calculates the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Convert confusion matrix to percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create a confusion matrix display
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=['Non', 'Very mild', 'Mild', 'Moderate'])

    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues, values_format='.2f', colorbar=False)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title('Confusion Matrix', fontsize=20)
    plt.xlabel('Predicted Label', fontsize=18, labelpad=16)
    plt.ylabel('True Label', fontsize=18)

    # Ajustar o tamanho da fonte dos números dentro das células
    for text in disp.text_.ravel():
        text.set_fontsize(18)

    # Ajustar a barra de cores (colorbar)
    cbar = plt.colorbar(disp.im_, ax=plt.gca())
    cbar.ax.tick_params(labelsize=18)  # Ajustar tamanho da fonte da barra de cores

    plt.tight_layout()  # Ajustar layout para evitar corte de labels

    if save_plot:
        plt.savefig('confusion_matrix.png')
    else:
        plt.show()



def plot_roc_curve(y_true, y_probs, class_names, save_plot=False):
    # Binarizing the labels
    y_true_binarized = label_binarize(y_true, classes=[0, 1, 2, 3])

    # Initializes dictionaries to store false positive rates (FPR), 
    # true positive rates (TPR), and area under the curve (AUC) for each class
    fpr = {}
    tpr = {}
    roc_auc = {}

    # Create a figure to plot the ROC curve
    plt.figure()

    # For each class, calculate the FPR and TPR curves and the AUC
    for i, class_name in enumerate(class_names):
        fpr[class_name], tpr[class_name], _ = roc_curve(y_true_binarized[:, i], np.array(y_probs)[:, i])
        roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])

        # Plota a curva ROC para a classe
        plt.plot(fpr[class_name], tpr[class_name], label=f'{class_name} (area = {roc_auc[class_name]:.2f})')

    # Plot the ROC curve for a random classifier
    plt.plot([0, 1], [0, 1], 'k--')

    # Defines the limits of the x and y axes
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # Define labels, title and legend
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title('Receiver Operating Characteristic', fontsize=14)
    plt.legend(loc="lower right")
    if save_plot:
        plt.savefig('roc_curve.png') 
    else:
        plt.show()


# Para gerar o histograma das classes, remover depois
# import seaborn as sns
# from collections import Counter
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from alzheimer_dataset import AlzheimerDataset
# batch_size          = 16


# # Configurar o estilo do Seaborn
# sns.set_theme(context="paper", style="white")

# # Obter os nomes das classes
# class_names = ['Non demented', 'Very mild demented', 'Mild demented', 'Moderate demented']

# # Inicializar um Counter para contar as labels de ambas as datasets
# class_counts = Counter()

# # Defining the data path
# train_data_path = "./Data/train"
# test_data_path  = "./Data/test"

# # Defining train transformations
# train_transform = transforms.Compose([
#     # transforms.Grayscale(num_output_channels=3),
#     transforms.Resize((224, 224)),
#     # np.array,
#     # get_mri_augmentation_sequence().augment_image,        # AUGMENTATION
#     # np.copy,
#     transforms.ToTensor(),
#     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # when using 3 channels
#     transforms.Normalize(mean=[0.485], std=[0.229])
# ])

# # Defining transformations
# test_transform = transforms.Compose([
#     # transforms.Grayscale(num_output_channels=3),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     transforms.Normalize(mean=[0.485], std=[0.229])
# ])

# # Loading training data
# train_dataset = AlzheimerDataset(train_data_path, transform=train_transform)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# # Loading training data
# test_dataset = AlzheimerDataset(test_data_path, transform=test_transform)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# # Contar as labels no dataset de treinamento
# for _, labels in train_loader:
#     class_counts.update(labels.tolist())

# # Contar as labels no dataset de teste
# for _, labels in test_loader:
#     class_counts.update(labels.tolist())

# # Converter os contadores para listas para plotagem
# class_indices, counts = zip(*sorted(class_counts.items(), key=lambda x: x[1], reverse=True))
# class_labels = [class_names[i] for i in class_indices]

# # Configurar a paleta de cores mais escura e inverter a ordem
# color = sns.color_palette("Blues", 4)[3]

# # Configurações adicionais de plotagem
# plt.figure(figsize=(12, 8))
# bars = plt.bar(class_labels, counts, align='center', color=color)
# plt.xlabel('Class Name', fontsize=18, labelpad=18)
# plt.ylabel('Number of Images', fontsize=18, labelpad=18)
# plt.xticks(rotation=0, fontsize=16)  # Rotacionar e ajustar o tamanho das labels
# plt.yticks(fontsize=16)
# plt.title('Number of Images per Class', fontsize=20)

# # Adicionar as quantidades exatas em cima de cada barra
# for bar, count in zip(bars, counts):
#     yval = bar.get_height()
#     plt.text(bar.get_x() + bar.get_width() / 2, yval, count, ha='center', va='bottom', fontsize=16)  # Adicionar o texto com a quantidade

# # Adicionar o grid para as barras e bordas ao redor do gráfico
# sns.despine(left=True, bottom=True)
# # plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# # plt.gca().spines['top'].set_visible(True)
# # plt.gca().spines['right'].set_visible(True)
# # plt.gca().spines['top'].set_color('black')
# # plt.gca().spines['right'].set_color('black')
# # plt.gca().spines['bottom'].set_color('black')
# # plt.gca().spines['left'].set_color('black')

# plt.show()