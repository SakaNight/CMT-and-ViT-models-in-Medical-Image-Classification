import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

def plot_training_history(history, title, save_dir=None):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], 'bo-', label='Training accuracy')
    plt.plot(history['val_accuracy'], 'b-', label='Validation accuracy')
    plt.title(f'Training and validation accuracy - {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], 'bo-', label='Training loss')
    plt.plot(history['val_loss'], 'b-', label='Validation loss')
    plt.title(f'Training and validation loss - {title}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if save_dir:
        plt.savefig(os.path.join(save_dir, f'training_history_{title}.png'))
    plt.show()

def plot_confusion_matrix(model, test_loader, true_labels, all_preds, save_dir=None, average=None):
    model.eval()

    true_labels = np.array(true_labels)
    cm = confusion_matrix(true_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.show()

    accuracy = accuracy_score(true_labels, all_preds)
    precision = precision_score(true_labels, all_preds, average=average)
    recall = recall_score(true_labels, all_preds, average=average)
    f1 = f1_score(true_labels, all_preds, average=average)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

def save_training_results(history, model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(save_dir, 'trained_model.pth'))

    plot_training_history(history, 'Model', save_dir)
