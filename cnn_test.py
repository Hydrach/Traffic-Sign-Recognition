import torch
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch import nn

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Test dataset
test_data = datasets.ImageFolder("dataset_split/test", transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

# Model yükle
model = resnet18()
model.fc = nn.Linear(model.fc.in_features, len(test_data.classes))
model.load_state_dict(torch.load("cnn_model.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"CNN Accuracy: %{accuracy:.2f}")