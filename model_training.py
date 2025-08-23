# model_training.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os
import json

# --- Configuration ---
data_dir = "synthetic_bin_dataset"  # Folder with subfolders: empty/, half_full/, full/
num_classes = 3
batch_size = 8
epochs = 3  # Lightweight training
lr = 0.001
model_path = "saved_model.pth"
labels_path = "labels.json"

# --- Dataset ---
class BinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.data = []
        self.labels = {}
        classes = sorted(os.listdir(root_dir))
        for idx, cls in enumerate(classes):
            self.labels[idx] = cls
            cls_folder = os.path.join(root_dir, cls)
            for img_file in os.listdir(cls_folder):
                self.data.append((os.path.join(cls_folder, img_file), idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# --- Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Load Dataset ---
dataset = BinDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Model ---
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)

# --- Loss & Optimizer ---
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# --- Training Loop ---
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

# --- Save Model ---
torch.save(model, model_path)
print(f"Model saved to {model_path}")

# --- Save Labels ---
with open(labels_path, "w") as f:
    json.dump(dataset.labels, f)
print(f"Labels saved to {labels_path}")
