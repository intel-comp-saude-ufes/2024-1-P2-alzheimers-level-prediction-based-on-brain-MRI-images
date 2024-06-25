import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # Convertendo para porcentagem
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='.2f')  # Formato para exibir porcentagem com duas casas decimais
    plt.show()


def plot_roc_curve(y_true, y_pred, class_names):
    # Binarizando os rótulos
    y_true_binarized = label_binarize(y_true, classes=[0, 1, 2, 3])
    y_pred_binarized = label_binarize(y_pred, classes=[0, 1, 2, 3])

    fpr = {}
    tpr = {}
    roc_auc = {}

    plt.figure()
    for i, class_name in enumerate(class_names):
        fpr[class_name], tpr[class_name], _ = roc_curve(y_true_binarized[:, i], y_pred_binarized[:, i])
        roc_auc[class_name] = auc(fpr[class_name], tpr[class_name])
        plt.plot(fpr[class_name], tpr[class_name], label=f'{class_name} (area = {roc_auc[class_name]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()