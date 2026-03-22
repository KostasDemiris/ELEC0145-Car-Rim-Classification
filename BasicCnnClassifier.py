import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
from thop import profile    

epochs = 15
batch_size = 6
image_size = 224
learning_rate = 1e-4

data_dir = 'Sorted Classification Dataset'
output_dir = 'runs/classify/CarRimClassifier_CNN'


class CNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.4),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

train_transforms = transforms.Compose([
    transforms.Resize((image_size + 32, image_size + 32)),
    transforms.RandomCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2),
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

model = CNN(num_classes=num_classes)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
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

    # FLOP count and parameter count for presentation
    model = CNN(10)
    dummy_input = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(dummy_input,))
    print(f"FLOPs: {flops / 1e9:.2f}G, Params: {params / 1e6:.2f}M")