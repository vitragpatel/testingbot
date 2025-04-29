import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset paths
train_dir = 'mo_db/train'
val_dir = 'mo_db/val'

# Load data
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Define model
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (INSIDE main)
def train_model(epochs=5):
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {running_loss / len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), 'model.pth')
    print("✅ Model saved to model.pth")

# ENTRY POINT FOR WINDOWS
if __name__ == '__main__':
    train_model()






# ###### run this  --- python train_model.py ----- to train the model once only =================

# import os
# from pathlib import Path
# import torch
# import torch.nn as nn
# from torchvision import datasets, models, transforms
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# # Paths
# DATA_DIR = Path("mo_db")
# MODEL_SAVE_PATH = Path("model.pth")
# print(f"Using data from ...............{DATA_DIR}")
# # Hyperparameters
# BATCH_SIZE = 32
# EPOCHS = 5
# LEARNING_RATE = 1e-4
# NUM_WORKERS = 0
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device ...............{DEVICE}")
# # Transforms
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225])
# ])

# print(f"Using batch size ...............{transform}")
# # Load Datasets
# train_dataset = datasets.ImageFolder(DATA_DIR / "train", transform=transform)
# val_dataset   = datasets.ImageFolder(DATA_DIR / "val", transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
# val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# print(f"Using train dataset ...............{train_dataset}")
# print(f"Using val dataset ...............{val_dataset}")
# print(f"Using train loader ...............{train_loader}")
# print(f"Using val loader ...............{val_loader}")

# # Model
# model = models.resnet18(pretrained=True)
# num_classes = len(train_dataset.classes)
# model.fc = nn.Linear(model.fc.in_features, num_classes)
# model = model.to(DEVICE)

# print(f"Using model ...............{model}")
# print(f"Using num_classes ...............{num_classes}")
# print(f"Using model.fc ...............{model.fc}")

# # Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# print("start training...................")
# # Training Loop
# loop_count = 0
# for epoch in range(EPOCHS):
    
#     if loop_count % 100 == 0:
#         print(f"Processing {epoch}...")
    
#     model.train()
#     total_loss = 0
#     print(f"Epoch {epoch+1}/{EPOCHS}")

#     for images, labels in tqdm(train_loader, desc="Training"):
#         images, labels = images.to(DEVICE), labels.to(DEVICE)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()

#     loop_count += 1
#     print(f"Train Loss: {total_loss / len(train_loader):.4f}")

#     # Validation
#     model.eval()
#     correct = total = 0
#     with torch.no_grad():
#         for images, labels in val_loader:
#             images, labels = images.to(DEVICE), labels.to(DEVICE)
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()

#     accuracy = 100 * correct / total
#     print(f"Validation Accuracy: {accuracy:.2f}%")

# print("Training complete!")
# # Save model
# torch.save({
#     "model_state_dict": model.state_dict(),
#     "class_to_idx": train_dataset.class_to_idx
# }, MODEL_SAVE_PATH)

# print(f"✅ Model saved to {MODEL_SAVE_PATH}")
