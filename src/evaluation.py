import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

def calculate_metrics(ground_truth, predictions):
    true_positive = sum((ground_truth == 1) & (predictions == 1))
    true_negative = sum((ground_truth == 0) & (predictions == 0))
    false_positive = sum((ground_truth == 0) & (predictions == 1))
    false_negative = sum((ground_truth == 1) & (predictions == 0))

    sensitivity = true_positive / (true_positive + false_negative)
    specificity = true_negative / (true_negative + false_positive)
    return sensitivity, specificity

def draw_roc_curve(ground_truth, predictions):
    fpr, tpr, _ = roc_curve(ground_truth, predictions)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
