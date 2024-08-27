import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
    def __init__(self, root_dir, transform=None, target_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_class = target_class
        self.file_paths = []
        self.labels = []

        if target_class is not None:
            class_dir = os.path.join(root_dir, target_class)
            if os.path.exists(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith('.png'):
                        self.file_paths.append(os.path.join(class_dir, file_name))
                        self.labels.append(target_class)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label



data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class BirdClassifierCNN(nn.Module):
    def __init__(self, num_classes):
        super(BirdClassifierCNN, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


def plot_confusion_matrix(cm, classes, title='Confusion matrix'):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    # Define the class you want to finetune
    finetune_class_index = 18  # For example, 18 corresponds to "Mourning Dove"
    finetune_class_name = bird_dict[finetune_class_index]
    num_classes = len(bird_dict)

    # Load the dataset for the specific class
    dataset = BirdDataset(root_dir='../mels_5_sec', transform=data_transforms, target_class=finetune_class_name)
    print(f"Total number of samples in the dataset: {len(dataset)}")

    # Load the existing model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BirdClassifierCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load('../bird_classifier_best_model.pth'))

    # Set the model to training mode
    model.train()

    # Create a weighted sampler for the finetuning class
    class_weights = torch.ones(num_classes)  # Default to equal weights
    class_weights[finetune_class_index] = 1.0
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    num_epochs = 50
    best_val_loss = float('inf')
    patience = 5
    early_stopping_counter = 0

    # Split dataset for validation
    dataset_size = len(dataset)
    val_size = int(0.2 * dataset_size)
    train_size = dataset_size - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss}, Accuracy: {accuracy:.2f}%')

        cm = confusion_matrix(all_labels, all_predictions)
        plot_confusion_matrix(cm, classes=[finetune_class_name])

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'bird_classifier_finetune.pth')
            print("Model saved to bird_classifier_finetune.pth")
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break

    print("Training completed")
