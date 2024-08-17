import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from model_vit_small import ViTClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from plot_utils import plot_training_history, plot_confusion_matrix, save_training_results
import time

def train_model(model, train_loader, val_loader, device, num_epochs=10):
    class_weights = torch.FloatTensor([0.5, 0.5, 0.5, 0.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    throughput_list = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            total_train += labels.size(0)          

        train_acc = train_correct / total_train
        val_loss, val_acc, val_true_labels, val_preds, throughput = evaluate_model(model, val_loader, device)
        throughput_list.append(throughput)

        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)
        history['loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}, Throughput: {throughput:.2f} images/sec")
    
    avg_throughput = sum(throughput_list) / len(throughput_list)
    return history, val_true_labels, val_preds, avg_throughput

def evaluate_model(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loss = 0
    correct = 0
    total = 0

    all_preds = []
    true_labels = []

    start_time = time.time()

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    end_time = time.time()
    eval_time = end_time - start_time
    throughput = total / eval_time

    val_loss /= len(data_loader)
    val_accuracy = correct / total
    precision = precision_score(true_labels, all_preds, average='macro', zero_division=1)
    recall = recall_score(true_labels, all_preds, average='macro', zero_division=1)
    f1 = f1_score(true_labels, all_preds, average='macro', zero_division=1)

    print(f"Accuracy: {val_accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    return val_loss, val_accuracy, true_labels, all_preds, throughput

def reshape_images(images):
    if images.shape[1] != 3:
        images = images.permute(0, 3, 1, 2)
    return images

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

    # Load COVID-19 data
    covid_train_images = reshape_images(torch.load('your_local_folder/data/augmented/COVID-19/X_train_covid_augmented.pt'))
    covid_train_labels = torch.load('your_local_folder/data/augmented/COVID-19/y_train_covid_augmented.pt')
    covid_val_images = reshape_images(torch.load('your_local_folder/data/local_test/COVID-19/val_images.pt'))
    covid_val_labels = torch.load('your_local_folder/data/local_test/COVID-19/val_labels.pt')

    print(f"COVID-19 train images shape: {covid_train_images.shape}")
    print(f"COVID-19 val images shape: {covid_val_images.shape}")

    covid_train_loader = DataLoader(TensorDataset(train_transform(covid_train_images), covid_train_labels), batch_size=32, shuffle=True)
    covid_val_loader = DataLoader(TensorDataset(covid_val_images, covid_val_labels), batch_size=32, shuffle=False)

    # Train and evaluate ViT model on COVID-19 data
    for attention_type in ['standard', 'sparse', 'local']:
        print(f"Training ViT model with {attention_type} attention on COVID-19 data")
        vit_model = ViTClassifier(num_classes=4, attention_type=attention_type)
        vit_model.to(device)
        history, val_true_labels, val_preds, avg_throughput = train_model(vit_model, covid_train_loader, covid_val_loader, device)
        save_training_results(history, vit_model, f'your_local_folder/results/ViTS_COVID_{attention_type}')
        plot_confusion_matrix(vit_model, covid_val_loader, val_true_labels, val_preds, save_dir=f'your_local_folder/results/ViT_COVID_{attention_type}', average='macro')
        print(f"Final model throughput: {avg_throughput:.2f} images/second")

    # Load NIH data
    nih_train_images = reshape_images(torch.load('your_local_folder/data/augmented/NIH/X_train_nih_augmented.pt'))
    nih_train_labels = torch.load('your_local_folder/data/augmented/NIH/y_train_nih_augmented.pt')
    nih_val_images = reshape_images(torch.load('your_local_folder/data/local_test/NIH/val_images.pt'))
    nih_val_labels = torch.load('your_local_folder/data/local_test/NIH/val_labels.pt')

    print(f"NIH train images shape: {nih_train_images.shape}")
    print(f"NIH val images shape: {nih_val_images.shape}")

    nih_train_loader = DataLoader(TensorDataset(train_transform(nih_train_images), nih_train_labels), batch_size=32, shuffle=True)
    nih_val_loader = DataLoader(TensorDataset(nih_val_images, nih_val_labels), batch_size=32, shuffle=False)

    # Train and evaluate ViT model on NIH data
    for attention_type in ['standard', 'sparse', 'local']:
        print(f"Training ViT model with {attention_type} attention on NIH data")
        vit_model = ViTClassifier(num_classes=4, attention_type=attention_type)
        vit_model.to(device)
        history, val_true_labels, val_preds, avg_throughput = train_model(vit_model, nih_train_loader, nih_val_loader, device)
        save_training_results(history, vit_model, f'your_local_folder/results/ViTS_NIH_{attention_type}')
        plot_confusion_matrix(vit_model, nih_val_loader, val_true_labels, val_preds, save_dir=f'your_local_folder/results/ViT_NIH_{attention_type}', average='macro')
        print(f"Final model throughput: {avg_throughput:.2f} images/second")

if __name__ == "__main__":
    main()
