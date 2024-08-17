import os
import numpy as np
import pandas as pd
import cv2
import torch
from sklearn.model_selection import train_test_split

def preprocess_covid_data(image_dir, metadata_file, label_value, img_size=224):
    data = pd.read_excel(metadata_file)
    data.columns = data.columns.str.strip()
    images = []
    labels = []
    for index, row in data.iterrows():
        img_path = os.path.join(image_dir, row['FILE NAME'] + '.png')
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        images.append(img)
        labels.append(label_value)
    return np.array(images), np.array(labels)

def sample_data_per_category(images, labels, samples_per_category):
    sampled_images = []
    sampled_labels = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        sampled_indices = np.random.choice(label_indices, samples_per_category, replace=False)
        sampled_images.append(images[sampled_indices])
        sampled_labels.append(labels[sampled_indices])
    return np.concatenate(sampled_images), np.concatenate(sampled_labels)

def save_data(images, labels, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(torch.tensor(images, dtype=torch.float32), os.path.join(output_dir, f'{prefix}_images.pt'))
    torch.save(torch.tensor(labels, dtype=torch.long), os.path.join(output_dir, f'{prefix}_labels.pt'))

def main():
    categories = [
        ('COVID-19', 'your_local_folder/dataset/COVID-19_Radiography_Dataset/COVID/images', 'D:/Study/Waterloo/CS686/project/dataset/COVID-19_Radiography_Dataset/COVID.metadata.xlsx', 0),
        ('Normal', 'your_local_folder/dataset/COVID-19_Radiography_Dataset/Normal/images', 'D:/Study/Waterloo/CS686/project/dataset/COVID-19_Radiography_Dataset/Normal.metadata.xlsx', 1),
        ('Lung_Opacity', 'your_local_folder/dataset/COVID-19_Radiography_Dataset/Lung_Opacity/images', 'D:/Study/Waterloo/CS686/project/dataset/COVID-19_Radiography_Dataset/Lung_Opacity.metadata.xlsx', 2),
        ('Viral_Pneumonia', 'your_local_folder/dataset/COVID-19_Radiography_Dataset/Viral_Pneumonia/images', 'D:/Study/Waterloo/CS686/project/dataset/COVID-19_Radiography_Dataset/Viral_Pneumonia.metadata.xlsx', 3)
    ]

    all_images = []
    all_labels = []

    samples_per_category = 250

    for category_name, image_dir, metadata_file, label_value in categories:
        print(f"Processing {category_name} images...")
        images, labels = preprocess_covid_data(image_dir, metadata_file, label_value, img_size=224)
        sampled_images, sampled_labels = sample_data_per_category(images, labels, samples_per_category)
        all_images.append(sampled_images)
        all_labels.append(sampled_labels)
    
    all_images = np.concatenate(all_images)
    all_labels = np.concatenate(all_labels)

    train_images, test_images, train_labels, test_labels = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    output_dir = 'your_local_folder/data/local_test/COVID-19'
    save_data(train_images, train_labels, output_dir, 'train')
    save_data(val_images, val_labels, output_dir, 'val')
    save_data(test_images, test_labels, output_dir, 'test')

    print("Data preprocessing complete. Saved to disk.")

if __name__ == "__main__":
    main()
