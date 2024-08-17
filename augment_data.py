import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if isinstance(image, torch.Tensor):
            image = image.numpy()
        if image.shape[0] == 3: 
            image = np.transpose(image, (1, 2, 0)) 
        if self.transform:
            image = self.transform(image)
        return image, label

def augment_data(images, labels, output_dir, prefix, batch_size=32):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor()
    ])

    dataset = CustomDataset(images, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    augmented_images = []
    augmented_labels = []

    for images, labels in dataloader:
        augmented_images.append(images)
        augmented_labels.append(labels)

    augmented_images = torch.cat(augmented_images)
    augmented_labels = torch.cat(augmented_labels)

    torch.save(augmented_images, os.path.join(output_dir, f'X_{prefix}_augmented.pt'))
    torch.save(augmented_labels, os.path.join(output_dir, f'y_{prefix}_augmented.pt'))

def main():
    covid_train_images = torch.load('your_local_folder/data/local_test/COVID-19/train_images.pt')
    covid_train_labels = torch.load('your_local_folder/data/local_test/COVID-19/train_labels.pt')
    nih_train_images = torch.load('your_local_folder/data/local_test/NIH/train_images.pt')
    nih_train_labels = torch.load('your_local_folder/data/local_test/NIH/train_labels.pt')

    augment_data(covid_train_images, covid_train_labels, 'your_local_folder/data/augmented/COVID-19', 'train_covid', batch_size=32)

    augment_data(nih_train_images, nih_train_labels, 'your_local_folder/data/augmented/NIH', 'train_nih', batch_size=32)

if __name__ == "__main__":
    main()
