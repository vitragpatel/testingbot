import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

# Device Setup
print("Setting up device...")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
print(f"Using device: {DEVICE}")

# Paths
print("Setting up paths...")
DATA_DIR = Path("/kaggle/input/zalando-store-crawl/zalando")
MODEL_SAVE_PATH = Path("/kaggle/working/resnet50_model.pth")
LOG_DIR = Path("/kaggle/working/logs_resnet50")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameters
print("Initializing hyperparameters...")
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 3e-4
NUM_WORKERS = 2
WEIGHT_DECAY = 1e-5
VALIDATION_SPLIT = 0.2
SEED = 42
EARLY_STOPPING_PATIENCE = 5
LR_SCHEDULER_PATIENCE = 2

torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed(SEED)

# Data Augmentation
print("Setting up data augmentation...")
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Data Loading and Splitting
print("Loading and splitting dataset...")
dataset = datasets.ImageFolder(DATA_DIR)
num_classes = len(dataset.classes)
train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print("Dataset loaded successfully.")

# Model Setup
print("Setting up the model...")
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

for param in model.parameters():
    param.requires_grad = False
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)
print("Model setup complete.")

# Training Setup
print("Setting up training components...")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=LR_SCHEDULER_PATIENCE, verbose=True, min_lr=1e-7)

print("Starting training loop...")
# Training Loop
best_accuracy = 0.0
epochs_no_improve = 0

for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}/{EPOCHS}...")
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]', leave=False):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    epoch_train_loss = running_loss / train_size

    # Validation Phase
    print("Validating...")
    model.eval()
    val_running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]', leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_preds += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    epoch_val_loss = val_running_loss / val_size
    epoch_val_accuracy = 100 * correct_preds / total_samples

    scheduler.step(epoch_val_accuracy)

    if epoch_val_accuracy > best_accuracy:
        best_accuracy = epoch_val_accuracy
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"New best model saved with accuracy: {best_accuracy:.2f}%")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print("Early stopping triggered.")
        break

print("Training complete.")

# Final Evaluation
print("Evaluating final model...")
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.eval()

correct_preds = 0
total_samples = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct_preds += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

final_accuracy = 100 * correct_preds / total_samples
print(f"Final Validation Accuracy: {final_accuracy:.2f}%")

# Classification Report
print("Generating classification report...")

# Confusion Matrix
print("Generating confusion matrix...")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
