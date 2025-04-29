import os
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split, Subset # Added Subset for type hints if needed
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns # For confusion matrix plotting
import numpy as np # For confusion matrix

# 1. Enhanced Device Setup ==============================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True  # Optimizes CUDA performance if input sizes don't vary much

# 2. Data & Model Paths ================================================
# --- !!! IMPORTANT: Ensure DATA_DIR points to the ROOT of your unzipped dataset !!! ---
# --- It should contain the class folders directly (e.g., /kaggle/input/test-db-sm/test_db/Denim, ...) ---
DATA_DIR = Path("/kaggle/input/zalando-store-crawl/zalando")
MODEL_SAVE_PATH = Path("/kaggle/working/test_modelv2.pth")
LOG_DIR = Path("/kaggle/working/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 3. Hyperparameters ===================================================
BATCH_SIZE = 32
EPOCHS = 4
# LEARNING_RATE = 1e-4
LEARNING_RATE = 3e-4
NUM_WORKERS = 2 # os.cpu_count() # Can use os.cpu_count(), but 2 is safer on Kaggle
WEIGHT_DECAY = 1e-5
VALIDATION_SPLIT = 0.2 # Use 20% of data for validation
SEED = 42 # For reproducible splits
EARLY_STOPPING_PATIENCE = 5 # Renamed from PATIENCE for clarity
LR_SCHEDULER_PATIENCE = 2 # Patience for reducing learning rate

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed(SEED)

print("Hyperparameters set.")

## ==============================================================================

# 4. Enhanced Data Augmentation ========================================
# Increased augmentation strength slightly
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), # Crop a bit more aggressively
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Added ColorJitter
    transforms.RandomRotation(15), # Added RandomRotation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
print("Transforms defined.")


## ==========================================================================

# 5. Data Loading and Splitting ======================================
# --- Simplified Data Loading ---
# Assume DATA_DIR contains class folders directly. Load everything first.
print(f"Loading dataset from: {DATA_DIR}")


# Create two instances of the dataset with different transforms
print("Creating dataset instances with respective transforms...")
dataset_for_train_split = datasets.ImageFolder(DATA_DIR, transform=train_transform)
dataset_for_val_split = datasets.ImageFolder(DATA_DIR, transform=val_transform)

# Ensure classes are the same in both instances (sanity check)
assert dataset_for_train_split.classes == dataset_for_val_split.classes, "Class mismatch between dataset instances!"
num_classes = len(dataset_for_train_split.classes)
class_names = dataset_for_train_split.classes
class_to_idx = dataset_for_train_split.class_to_idx
print(f"Found {len(dataset_for_train_split)} images in {num_classes} classes.")
print(f"Classes: {class_names}")

# Calculate split sizes
dataset_size = len(dataset_for_train_split)
val_size = int(VALIDATION_SPLIT * dataset_size)
train_size = dataset_size - val_size

# Split indices (important: use a generator for reproducibility)
print(f"Splitting dataset: Train={train_size}, Validation={val_size}")
indices = list(range(dataset_size))
# No need to shuffle here if DataLoader shuffles; random_split does it internally but seed helps
train_indices, val_indices = random_split(indices, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))

# Create Subset datasets using the *correct* transforms
train_dataset = Subset(dataset_for_train_split, train_indices)
val_dataset = Subset(dataset_for_val_split, val_indices)

print("Creating DataLoaders...")
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    # drop_last=True # Removed for now, usually not necessary unless causing issues
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, # No need to shuffle validation data
    num_workers=NUM_WORKERS,
    pin_memory=True
)

# Class distribution analysis (on the training set)
print("Analyzing class distribution in the training set...")
train_class_counts = torch.zeros(num_classes)
# Iterate through the train_dataset Subset to get labels
for idx in train_dataset.indices:
    label = dataset_for_train_split.targets[idx] # Access targets from the original dataset
    train_class_counts[label] += 1

print("Training Set Class distribution:")
plt.figure(figsize=(10, 5))
plt.bar(class_names, train_class_counts.numpy())
plt.xticks(rotation=60, ha='right') # Rotate more for potentially many classes
plt.title("Training Set Class Distribution")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig(LOG_DIR / 'class_distribution.png')
plt.show()
print("Data loading and splitting complete.")

##===========================================================================

# 6. Enhanced Model Setup ==============================================
print("Setting up the model...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # Use updated weights API

# Replace final layer (num_classes is derived above)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.BatchNorm1d(512), # Added BatchNorm for stability
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

# Freeze early layers (fine-tune layer 3, layer 4 and fc)
print("Freezing initial layers, unfreezing layer3, layer4 and fc...")
for param in model.parameters():
    param.requires_grad = False
# Unfreeze layer 3
for param in model.layer3.parameters():
    param.requires_grad = True
# Unfreeze layer 4
for param in model.layer4.parameters():
    param.requires_grad = True
# Unfreeze the fully connected layer
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)
print("Model setup complete.")

# --- Optional: Print number of trainable parameters ---
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

## =========================================================================

# 7. Enhanced Training Setup ===========================================
print("Setting up loss, optimizer, and scheduler...")
# Consider adding label smoothing if needed for very confident predictions
criterion = nn.CrossEntropyLoss() # Add label_smoothing=0.1 maybe

# Filter parameters that require gradients for the optimizer
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)

scheduler = ReduceLROnPlateau(optimizer,
                             mode='max',
                             factor=0.5,
                             patience=LR_SCHEDULER_PATIENCE, # Use specific patience
                             verbose=True,
                             min_lr=1e-6)

# Tracking variables
best_accuracy = 0.0
epochs_no_improve = 0 # Counter for early stopping
train_losses = []
val_accuracies = []
val_losses = []

print("Training setup complete.")

## =========================================================================

# 8. Training Loop with Early Stopping =================================
print("Starting training loop...")
for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    # Use train_size which is calculated correctly now
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]', unit='batch')

    for images, labels in progress_bar:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True) # Use non_blocking for potential speedup

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping (optional but can help)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item() * images.size(0)

        # Update progress bar
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')

    # Use train_size calculated earlier
    epoch_train_loss = running_loss / train_size
    train_losses.append(epoch_train_loss)
    progress_bar.close() # Close the training progress bar

    # --- Validation Phase ---
    model.eval()
    val_running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]', unit='batch')
    with torch.no_grad():
        for images, labels in val_progress_bar:
            images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_preds += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            val_progress_bar.set_postfix(acc=f'{(100 * correct_preds / total_samples):.2f}%')

    val_progress_bar.close() # Close the validation progress bar

    # Use val_size calculated earlier
    epoch_val_loss = val_running_loss / val_size
    epoch_val_accuracy = 100 * correct_preds / total_samples
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy)

    print(f'\nEpoch {epoch+1} Summary:')
    print(f'Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Accuracy: {epoch_val_accuracy:.2f}%')

    # Learning rate scheduling
    scheduler.step(epoch_val_accuracy)

    # --- Save Best Model ---
    if epoch_val_accuracy > best_accuracy:
        print(f'Validation accuracy improved ({best_accuracy:.2f}% -> {epoch_val_accuracy:.2f}%). Saving model...')
        best_accuracy = epoch_val_accuracy
        epochs_no_improve = 0 # Reset counter
        # class_to_idx is available directly from dataset_for_train_split
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
            'class_to_idx': class_to_idx # Save the mapping
        }, MODEL_SAVE_PATH)
    else:
        epochs_no_improve += 1
        print(f'Validation accuracy did not improve for {epochs_no_improve} epoch(s).')

    # # --- Early Stopping Check ---
    # if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
    #     print(f'\nEarly stopping triggered after {epoch + 1} epochs.')
    #     break

print("\nTraining finished.")


## =========================================================================


# 9. Final Evaluation =================================================
print("Loading best model for final evaluation...")
# Load the best model state
checkpoint = torch.load(MODEL_SAVE_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval() # Ensure model is in eval mode

print("\nClassification Report (from last validation epoch where best model was saved or training ended):")
# Ensure class_names aligns with the labels used in the report
print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

# Plot Confusion Matrix
print("Plotting Confusion Matrix...")
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(max(10, num_classes // 2), max(8, num_classes // 2.5))) # Adjust size based on num_classes
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(LOG_DIR / 'confusion_matrix.png')
plt.show()


# Plot training history
print("Plotting Training History...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.axhline(best_accuracy, color='r', linestyle='--', label=f'Best Acc: {best_accuracy:.2f}%') # Mark best accuracy
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(LOG_DIR / 'training_history.png')
plt.show()

print(f"\n✅ Best validation accuracy achieved: {best_accuracy:.2f}%")
print(f"Model saved to: {MODEL_SAVE_PATH}")
print(f"Logs and plots saved to: {LOG_DIR}")
print("\nScript finished.")