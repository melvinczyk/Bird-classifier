import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np


class BirdClassifierCNN(nn.Module):
    def __init__(self, num_classes=29):
        super(BirdClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def main():
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = ImageFolder(root='mels', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = ImageFolder(root='mels', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BirdClassifierCNN(num_classes=29)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        class_accuracies = []
        for class_idx in range(29):
            class_mask = (all_labels == class_idx)
            class_acc = accuracy_score(all_labels[class_mask], all_preds[class_mask])
            class_accuracies.append(class_acc)

        average_accuracy = np.mean(class_accuracies)

        print(f'Epoch {epoch + 1}/{num_epochs}')
        print(f'Loss: {epoch_loss:.4f}')
        for class_idx, class_acc in enumerate(class_accuracies):
            print(f'Class {class_idx} Accuracy: {class_acc:.4f}')

        if average_accuracy > best_accuracy:
            best_accuracy = average_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Best model saved with accuracy: {best_accuracy:.4f}')


if __name__ == "__main__":
    main()
