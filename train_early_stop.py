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
        self.model = models.resnet18()
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
    plt.savefig('temp.png')

if __name__ == "__main__":
    dataset = BirdDataset(root_dir='mels_5_sec', transform=data_transforms)
    print(f"Total number of samples in the dataset: {len(dataset)}")

    class_counts = np.bincount(dataset.labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[dataset.labels]

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, stratify=dataset.labels, random_state=42)
    print(f"Number of training samples: {len(train_idx)}, Number of validation samples: {len(val_idx)}")

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_weights = [sample_weights[i] for i in train_idx]
    train_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(train_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=32, sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BirdClassifierCNN(num_classes=len(dataset.classes)).to(device)

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    num_epochs = 50
    best_val_loss = float('inf')
    patience = 5
    early_stopping_counter = 0

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
        class_correct = [0] * len(dataset.classes)
        class_total = [0] * len(dataset.classes)
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
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_correct[label] += (predicted[i] == labels[i]).item()
                    class_total[label] += 1
                    all_labels.append(label)
                    all_predictions.append(predicted[i].item())

        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        print(f'Validation Loss: {val_loss}, Accuracy: {accuracy:.2f}%')

        cm = confusion_matrix(all_labels, all_predictions)
        plot_confusion_matrix(cm, classes=dataset.classes)

        for i in range(len(dataset.classes)):
            if class_total[i] > 0:
                class_accuracy = 100 * class_correct[i] / class_total[i]
            else:
                class_accuracy = 0.0
            print(f'Accuracy of {dataset.classes[i]} : {class_accuracy:.2f}%')

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_model_early_stop_v2.pth')
            print("Model saved to best_model_early_stop_v2.pth")
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping triggered")
            break

    print("Training completed")
