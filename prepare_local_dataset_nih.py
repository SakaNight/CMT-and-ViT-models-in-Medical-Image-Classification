import os
import numpy as np
import pandas as pd
import cv2
import torch
from sklearn.model_selection import train_test_split

def preprocess_nih_data(image_dir, labels_file, img_size=224):
    data = pd.read_csv(labels_file)
    data.columns = data.columns.str.strip()
    
    images = []
    labels = []
    
    no_finding_images = []
    no_finding_labels = []
    
    disease_images = []
    disease_labels = []
    
    for index, row in data.iterrows():
        img_path = os.path.join(image_dir, row['Image Index'])
        
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        img = img / 255.0
        
        if row['Finding Labels'] == 'No Finding':
            no_finding_images.append(img)
            no_finding_labels.append(0)
        else:
            disease_images.append(img)
            disease_labels.append(1)
    
    no_finding_indices = np.random.choice(len(no_finding_images), 500, replace=False)
    disease_indices = np.random.choice(len(disease_images), 500, replace=False)
    
    images.extend([no_finding_images[i] for i in no_finding_indices])
    labels.extend([no_finding_labels[i] for i in no_finding_indices])
    images.extend([disease_images[i] for i in disease_indices])
    labels.extend([disease_labels[i] for i in disease_indices])
    
    return np.array(images), np.array(labels)

def save_data(images, labels, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(torch.tensor(images, dtype=torch.float32), os.path.join(output_dir, f'{prefix}_images.pt'))
    torch.save(torch.tensor(labels, dtype=torch.long), os.path.join(output_dir, f'{prefix}_labels.pt'))

def main():
    image_dirs = 'your_local_folder/dataset/NIH_Chest_X-ray_Dataset/images'
    labels_file = 'your_local_folder/dataset/NIH_Chest_X-ray_Dataset/sample_labels.csv'
    
    images, labels = preprocess_nih_data(image_dirs, labels_file, img_size=224)
    
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    output_dir = 'your_local_folder/data/local_test/NIH'
    save_data(train_images, train_labels, output_dir, 'train')
    save_data(val_images, val_labels, output_dir, 'val')
    save_data(test_images, test_labels, output_dir, 'test')

    print("Data preprocessing complete. Saved to disk.")

if __name__ == "__main__":
    main()