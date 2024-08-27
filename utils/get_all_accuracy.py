import os
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from sklearn.model_selection import train_test_split

# Bird dictionary
bird_dict = {
    0: "American Crow",
    1: "American Goldfinch",
    2: "American Robin",
    3: "Barred Owl",
    4: "Blue Jay",
    5: "Brown-headed Nuthatch",
    6: "Carolina Chickadee",
    7: "Carolina Wren",
    8: "Cedar Waxwing",
    9: "Chipping Sparrow",
    10: "Dark-eyed Junco",
    11: "Downy Woodpecker",
    12: "Eastern Bluebird",
    13: "Eastern Kingbird",
    14: "Eastern Phoebe",
    15: "Eastern Towhee",
    16: 'Empty',
    17: "House Finch",
    18: "Mourning Dove",
    19: "Myrtle Warbler",
    20: "Northern Cardinal",
    21: "Northern Flicker",
    22: "Northern Mockingbird",
    23: "Pine Warbler",
    24: "Purple Finch",
    25: "Red-bellied Woodpecker",
    26: "Red-winged Blackbird",
    27: "Song Sparrow",
    28: "Tufted Titmouse",
    29: "White-breasted Nuthatch",
}

class BirdDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.file_paths = []
        self.labels = []

        for label, bird_class in enumerate(self.classes):
            class_dir = os.path.join(root_dir, bird_class)
            for file_name in os.listdir(class_dir):
                if file_name.endswith('.png'):
                    self.file_paths.append(os.path.join(class_dir, file_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the model class
class BirdClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(BirdClassifierCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# Function to plot the confusion matrix
def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    # Load the validation set (using the same transformations as during training)
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Assuming 'mels_5_sec' is the root directory
    dataset = BirdDataset(root_dir='../mels_5_sec', transform=data_transforms)

    # Recreate or load validation indices
    if os.path.exists('val_idx.npy'):
        val_idx = np.load('val_idx.npy')
    else:
        train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.labels, random_state=42)
        np.save('val_idx.npy', val_idx)

    val_set = Subset(dataset, val_idx)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    # Load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BirdClassifierCNN(num_classes=len(dataset.classes)).to(device)
    model.load_state_dict(torch.load('../bird_classifier_best_model.pth'))
    model.eval()  # Set to evaluation mode

    # Initialize variables to track accuracy per class
    class_correct = [0] * len(dataset.classes)
    class_total = [0] * len(dataset.classes)
    all_labels = []
    all_predictions = []

    # Run inference
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
                all_labels.append(label)
                all_predictions.append(predicted[i].item())

    # Calculate and print the accuracy for each class
    for i in range(len(dataset.classes)):
        if class_total[i] > 0:
            class_accuracy = 100 * class_correct[i] / class_total[i]
        else:
            class_accuracy = 0.0
        print(f'Accuracy of {dataset.classes[i]} : {class_accuracy:.2f}%')

    # Generate and plot confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plot_confusion_matrix(cm, classes=dataset.classes)
