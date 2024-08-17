import torch
from data_loader import load_data, create_data_loaders

def print_label_distribution(labels, dataset_name):
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"{dataset_name} Label Distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label.item()}: {count.item()} samples")

def test_data_loaders():
    covid_train_images_path = 'your_local_folder/data/augmented/COVID-19/X_train_covid_augmented.pt'
    covid_train_labels_path = 'your_local_folder/data/augmented/COVID-19/y_train_covid_augmented.pt'
    covid_val_images_path = 'your_local_folder/data/local_test/COVID-19/val_images.pt'
    covid_val_labels_path = 'your_local_folder/data/local_test/COVID-19/val_labels.pt'
    
    nih_train_images_path = 'your_local_folder/data/augmented/NIH/X_train_nih_augmented.pt'
    nih_train_labels_path = 'your_local_folder/data/augmented/NIH/y_train_nih_augmented.pt'
    nih_val_images_path = 'your_local_folder/data/local_test/NIH/val_images.pt'
    nih_val_labels_path = 'your_local_folder/data/local_test/NIH/val_labels.pt'

    covid_train_images, covid_train_labels = load_data(covid_train_images_path, covid_train_labels_path)
    covid_val_images, covid_val_labels = load_data(covid_val_images_path, covid_val_labels_path)
    nih_train_images, nih_train_labels = load_data(nih_train_images_path, nih_train_labels_path)
    nih_val_images, nih_val_labels = load_data(nih_val_images_path, nih_val_labels_path)

    covid_train_loader, covid_val_loader = create_data_loaders(
        covid_train_images_path, covid_train_labels_path, 
        covid_val_images_path, covid_val_labels_path
    )
    nih_train_loader, nih_val_loader = create_data_loaders(
        nih_train_images_path, nih_train_labels_path, 
        nih_val_images_path, nih_val_labels_path
    )

    print(f"COVID-19 Train Loader: {len(covid_train_loader)} batches")
    print(f"COVID-19 Val Loader: {len(covid_val_loader)} batches")
    print(f"NIH Train Loader: {len(nih_train_loader)} batches")
    print(f"NIH Val Loader: {len(nih_val_loader)} batches")

    for batch in covid_train_loader:
        images, labels = batch
        print(f"COVID-19 Train Batch - images: {images.shape}, labels: {labels.shape}")
        break

    for batch in nih_train_loader:
        images, labels = batch
        print(f"NIH Train Batch - images: {images.shape}, labels: {labels.shape}")
        break

    print_label_distribution(covid_train_labels, "COVID-19 Train")
    print_label_distribution(covid_val_labels, "COVID-19 Val")
    print_label_distribution(nih_train_labels, "NIH Train")
    print_label_distribution(nih_val_labels, "NIH Val")
    
if __name__ == "__main__":
    test_data_loaders()
