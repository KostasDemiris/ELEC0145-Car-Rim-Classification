import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

EfficientNet_variant = 'efficientnet_b0' # It goes up to b7 
data_dir = 'Sorted Classification Dataset'
output_dir = 'runs/classify/CarRimClassifier_efficientnet'

# We're originally running this on mac...
device      = (
    'mps'  if torch.backends.mps.is_available() else
    'cuda' if torch.cuda.is_available()          else
    'cpu'
)
pin_mem  = device == 'cuda'
num_workers = 0 


class_num = 10
epochs = 25
image_size = 224
batch_size = 6

lr_0, lr_f = 1e-3, 1e-5
mixup_alpha = 0.0
flip_lr = 0.0


train_transforms = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=flip_lr),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.RandomRotation(degrees=15),
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_mem)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

model = timm.create_model(EfficientNet_variant, pretrained=True, num_classes=class_num)
model = model.to(device)

# Just matching the yolo mixup aug
def mixup_data(x, y, alpha=mixup_alpha):
    if alpha <= 0:
        return x, y, y, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# These mirror the YOLO learning rate scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr_0, momentum=0.937, weight_decay=5e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_0 * lr_f)

# Training
os.makedirs(output_dir, exist_ok=True)
best_acc  = 0.0
best_path = os.path.join(output_dir, 'best.pt')

for epoch in range(epochs):
    model.train()
    train_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Mixup augmentation
        images, targets_a, targets_b, lam = mixup_data(images, labels)

        optimizer.zero_grad()
        outputs = model(images)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += (lam * predicted.eq(targets_a).sum().item()
                    + (1 - lam) * predicted.eq(targets_b).sum().item())
        total += labels.size(0)

    scheduler.step()

    train_acc = correct / total

    # Eval
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            val_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total   += labels.size(0)

    val_acc = val_correct / val_total
    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch {epoch+1} of {epochs} has: train loss {train_loss/len(train_loader):.4f}, \
          train accuracy {train_acc:.4%}, and val accuracy {val_acc:.4%}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), best_path)
        print(f"  ✔ New best saved ({best_acc:.2%})")

# Test
model.load_state_dict(torch.load(best_path, map_location=device))
model.eval()

test_correct, test_total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        test_correct += predicted.eq(labels).sum().item()
        test_total   += labels.size(0)

test_acc = test_correct / test_total
print(f"\nTest Accuracy (Top-1): {test_acc:.2%}")