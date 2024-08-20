import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

test_dataset = ImageFolder(root='mels', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = BirdClassifierCNN(num_classes=29)
model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
model.eval()

all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

cm = confusion_matrix(all_labels, all_preds)

accuracy = accuracy_score(all_labels, all_preds)
print(f'Total Model Accuracy: {accuracy * 100:.2f}%')

class_names = [
    "AmericanCrow", "AmericanGoldfinch", "AmericanRobin", "BarredOwl", "BlueJay",
    "Brown-headedNuthatch", "CarolinaChickadee", "CarolinaWren", "CedarWaxwing", "ChippingSparrow",
    "Dark-eyedJunco", "DownyWoodpecker", "EasternBluebird", "EasternKingbird", "EasternPhoebe",
    "EasternTowhee", "Empty", "HouseFinch", "MourningDove", "MyrtleWarbler", "NorthernCardinal",
    "NorthernFlicker", "NorthernMockingbird", "PineWarbler", "PurpleFinch", "Red-belliedWoodpecker",
    "Red-wingedBlackbird", "SongSparrow", "TuftedTitmouse", "White-breastedNuthatch"
]

plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
