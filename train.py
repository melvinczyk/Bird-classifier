import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from collections import Counter
import os


class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(51136, 50)
        self.fc2 = nn.Linear(50, 30)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, Y) in enumerate(dataloader):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, Y).item()
            correct += (pred.argmax(1) == Y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f'\nTest Error:\nacc: {(100 * correct):>0.1f}%, avg loss: {test_loss:>8f}\n')

if __name__ == '__main__':
    dataset_path = './mels/'
    transform = transforms.Compose([transforms.Resize((201,81)), transforms.ToTensor()])

    bird_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    print(bird_dataset)

    class_map = bird_dataset.class_to_idx
    print("\nClass category and index of the images: {}\n".format(class_map))

    train_size = int(0.8 * len(bird_dataset))
    test_size = len(bird_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(bird_dataset, [train_size, test_size])

    print("Training size:", len(train_dataset))
    print("Testing size:", len(test_dataset))

    train_classes = [label for _, label in train_dataset]
    print(Counter(train_classes))

    train_dataloader = DataLoader(train_dataset, batch_size=15, num_workers=8, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=15, num_workers=8, shuffle=True, pin_memory=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    model = CNNet().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 100
    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print('completed')

    print(summary(model, input_size=(15, 3, 201, 81)))

    model.eval()
    class_map = {v: k for k, v in bird_dataset.class_to_idx.items()}
    with torch.no_grad():
        for batch, (X, Y) in enumerate(test_dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            print("Predicted:\nvalue={}, class_name= {}\n".format(pred[0].argmax(0), class_map[pred[0].argmax(0).item()]))
            print("Actual:\nvalue={}, class_name= {}\n".format(Y[0], class_map[Y[0].item()]))
            break
    torch.save(model.state_dict(), 'bird_classifier.pt')
