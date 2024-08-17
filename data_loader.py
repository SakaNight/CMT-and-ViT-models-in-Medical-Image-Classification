import torch
from torch.utils.data import DataLoader, TensorDataset

def load_data(images_path, labels_path):
    images = torch.load(images_path)
    labels = torch.load(labels_path)
    return images, labels

def create_data_loader(images, labels, batch_size=32):
    dataset = TensorDataset(images, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def create_data_loaders(train_images_path, train_labels_path, val_images_path, val_labels_path, batch_size=32):
    train_images, train_labels = load_data(train_images_path, train_labels_path)
    val_images, val_labels = load_data(val_images_path, val_labels_path)

    train_loader = create_data_loader(train_images, train_labels, batch_size)
    val_loader = create_data_loader(val_images, val_labels, batch_size)
    return train_loader, val_loader

if __name__ == "__main__":
    covid_train_images_path = 'your_local_folder/data/augmented/COVID-19/X_train_covid_augmented.pt'
    covid_train_labels_path = 'your_local_folder/data/augmented/COVID-19/y_train_covid_augmented.pt'
    covid_val_images_path = 'your_local_folder/data/local_test/COVID-19/val_images.pt'
    covid_val_labels_path = 'your_local_folder/data/local_test/COVID-19/val_labels.pt'

    nih_train_images_path = 'your_local_folder/data/augmented/NIH/X_train_nih_augmented.pt'
    nih_train_labels_path = 'your_local_folder/data/augmented/NIH/y_train_nih_augmented.pt'
    nih_val_images_path = 'your_local_folder/data/local_test/NIH/val_images.pt'
    nih_val_labels_path = 'your_local_folder/data/local_test/NIH/val_labels.pt'

    covid_train_loader, covid_val_loader = create_data_loaders(
        covid_train_images_path, covid_train_labels_path, 
        covid_val_images_path, covid_val_labels_path
    )
    
    print(f"COVID-19 Train Loader: {len(covid_train_loader)} batches")
    print(f"COVID-19 Val Loader: {len(covid_val_loader)} batches")

    nih_train_loader, nih_val_loader = create_data_loaders(
        nih_train_images_path, nih_train_labels_path, 
        nih_val_images_path, nih_val_labels_path
    )

    print(f"NIH Train Loader: {len(nih_train_loader)} batches")
    print(f"NIH Val Loader: {len(nih_val_loader)} batches")
