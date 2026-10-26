import torch
from torchvision import datasets, transforms
from torch import nn, optim
from torchvision.models import resnet18, ResNet18_Weights

# 🔥 DATA AUGMENTATION (çok önemli)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])

# 🔥 DATASET
train_data = datasets.ImageFolder("dataset_split/train", transform=transform)
val_data = datasets.ImageFolder("dataset_split/val", transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)

# 🔥 MODEL (GÜÇLÜ)
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(train_data.classes))

# 🔥 DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 🔥 LOSS + OPTIMIZER
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🔥 EPOCH (EN İYİ DENGE)
epochs = 10

train_losses = []
val_losses = []

for epoch in range(epochs):
    # 🔵 TRAIN
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    # 🔴 VALIDATION
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

# 🔥 MODELİ KAYDET
torch.save(model.state_dict(), "cnn_model.pth")

print("✅ CNN eğitim tamamlandı ve kaydedildi!")