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
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
    
    # Plot the confusion matrix
    disp.plot(cmap=plt.cm.Blues, values_format='.2f')
    plt.xticks(rotation=90)
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
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    if save_plot:
        plt.savefig('roc_curve.png')
    else:
        plt.show()