import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os

epochs = 25
batch_size = 6
image_size = 224
learning_rate = 1e-4

data_dir = 'Sorted Classification Dataset'
output_dir = 'runs/classify/CarRimClassifier_CNN'

class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

train_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transforms)
val_dataset   = datasets.ImageFolder(os.path.join(data_dir, 'val'),   transform=val_transforms)
test_dataset  = datasets.ImageFolder(os.path.join(data_dir, 'test'),  transform=val_transforms)

num_classes = len(train_dataset.classes) 

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

model = BasicCNN(num_classes=num_classes)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    best_acc  = 0.0
    best_path = os.path.join(output_dir, 'best.pt')

    for epoch in range(epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        for batch_idx, (images, labels) in enumerate(train_loader):

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total   += labels.size(0)

        train_acc  = correct / total
        current_lr = optimizer.param_groups[0]['lr']

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                val_loss    += loss_func(outputs, labels).item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1} of {epochs} has: train loss {train_loss/len(train_loader):.4f}, \
          train accuracy {train_acc:.4%}, and val accuracy {val_acc:.4%}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ✔ New best saved ({best_acc:.2%})")

    model.load_state_dict(torch.load(best_path))
    model.eval()

    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            _, predicted = model(images).max(1)
            test_correct += predicted.eq(labels).sum().item()
            test_total   += labels.size(0)

    print(f"\nTest Accuracy (Top-1): {test_correct / test_total:.2%}")