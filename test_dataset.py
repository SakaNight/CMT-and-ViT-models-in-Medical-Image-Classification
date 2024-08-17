import torch
import matplotlib.pyplot as plt

def print_label_distribution(labels, dataset_name):
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"{dataset_name} Label Distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"Label {label.item()}: {count.item()} samples")

def visualize_images(images, labels, title, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
    fig.suptitle(title)
    for i, ax in enumerate(axes):
        image = images[i].permute(1, 2, 0).numpy()
        ax.imshow(image)
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis('off')
    plt.show()

def load_and_check_augmented_data(covid_augmented_data_path, nih_augmented_data_path):
    augmented_covid_images = torch.load(covid_augmented_data_path + 'X_train_covid_augmented.pt')
    augmented_covid_labels = torch.load(covid_augmented_data_path + 'y_train_covid_augmented.pt')
    augmented_nih_images = torch.load(nih_augmented_data_path + 'X_train_nih_augmented.pt')
    augmented_nih_labels = torch.load(nih_augmented_data_path + 'y_train_nih_augmented.pt')

    print(f"Augmented COVID-19 images shape: {augmented_covid_images.shape}")
    print(f"Augmented COVID-19 labels shape: {augmented_covid_labels.shape}")
    print(f"Augmented NIH images shape: {augmented_nih_images.shape}")
    print(f"Augmented NIH labels shape: {augmented_nih_labels.shape}")

    visualize_images(augmented_covid_images, augmented_covid_labels, "Augmented COVID-19 Dataset Samples")
    visualize_images(augmented_nih_images, augmented_nih_labels, "Augmented NIH Dataset Samples")

    print_label_distribution(augmented_covid_labels, "Augmented COVID-19")
    print_label_distribution(augmented_nih_labels, "Augmented NIH")

def main():
    covid_augmented_data_path = 'your_local_folder/data/augmented/COVID-19/'
    nih_augmented_data_path = 'your_local_folder/data/augmented/NIH/'

    load_and_check_augmented_data(covid_augmented_data_path, nih_augmented_data_path)

if __name__ == "__main__":
    main()