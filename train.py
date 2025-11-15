
# TRAINING SCRIPT (VS CODE READY)
import os, json, numpy as np
from collections import defaultdict
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

DATA_DIR = "dataset"
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 5
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 42
CKPT_PATH = "animal_classifier.ckpt"

np.random.seed(SEED)
torch.manual_seed(SEED)

train_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])
valid_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

full_val = datasets.ImageFolder(DATA_DIR, transform=valid_tfms)
class_to_idx = full_val.class_to_idx
num_classes = len(class_to_idx)
assert num_classes >= 2, "Only one class found. Check dataset folder."

idx_by_class = defaultdict(list)
for i, (_, y) in enumerate(full_val.samples):
    idx_by_class[y].append(i)

train_idx, val_idx = [], []
for y, idxs in idx_by_class.items():
    idxs = np.array(idxs)
    np.random.shuffle(idxs)
    k = int(len(idxs)*(1-VAL_SPLIT))
    train_idx += idxs[:k].tolist()
    val_idx += idxs[k:].tolist()

train_ds = datasets.ImageFolder(DATA_DIR, transform=train_tfms)
val_ds   = full_val

train_loader = DataLoader(Subset(train_ds, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(Subset(val_ds, val_idx), batch_size=BATCH_SIZE, shuffle=False)

print("Classes:", num_classes)
print("Train images:", len(train_idx), "| Val images:", len(val_idx))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for p in model.parameters(): p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)

best_acc = 0
for epoch in range(1, EPOCHS+1):
    model.train()
    correct, total_loss = 0, 0
    for x,y in train_loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1)==y).sum().item()
    train_acc = correct / len(train_idx)

    model.eval()
    vcorrect, vloss = 0, 0
    with torch.no_grad():
        for x,y in val_loader:
            x,y = x.to(device), y.to(device)
            out = model(x)
            vloss += criterion(out,y).item()
            vcorrect += (out.argmax(1)==y).sum().item()
    val_acc = vcorrect / len(val_idx)

    print(f"Epoch {epoch}/{EPOCHS} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            "state_dict": model.state_dict(),
            "num_classes": num_classes,
            "class_to_idx": class_to_idx,
            "img_size": IMG_SIZE
        }, CKPT_PATH)

print("Training complete. Best accuracy:", best_acc)
