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
    print(f"CUDA Device Name: {torch.cuda.get_device_name(DEVICE)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(DEVICE)}")
    # Set benchmark true if input size is constant, potentially faster
    # If input sizes vary (e.g. RandomResizedCrop), setting it false might be safer
    torch.backends.cudnn.benchmark = True

# 2. Data & Model Paths ================================================
# --- !!! IMPORTANT: Ensure DATA_DIR points to the ROOT of your unzipped dataset !!! ---
# --- It should contain the class folders directly (e.g., /kaggle/input/test-db-sm/test_db/Denim, ...) ---
DATA_DIR = Path("/kaggle/input/zalando-store-crawl/zalando")
MODEL_SAVE_PATH = Path("/kaggle/working/resnet50_model.pth") # Changed model name
LOG_DIR = Path("/kaggle/working/logs_resnet50") # Changed log dir name
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 3. Hyperparameters ===================================================
# !!! ATTENTION: ResNet50 uses significantly more memory than ResNet18. !!!
# !!! You MAY need to DECREASE BATCH_SIZE if you get CUDA Out-Of-Memory errors. !!!
# Start with 32, try 16 or 8 if needed.
BATCH_SIZE = 16
EPOCHS = 4 # Rely on Early Stopping to find the optimal number
# LEARNING_RATE = 1e-4 # Alternative LR
LEARNING_RATE = 3e-4 # Good starting point for AdamW + fine-tuning
NUM_WORKERS = 2 # os.cpu_count() # Using 2 as a safe default on Kaggle
WEIGHT_DECAY = 1e-5 # Regularization term for AdamW
VALIDATION_SPLIT = 0.2 # Use 20% of data for validation
SEED = 42 # For reproducible splits
EARLY_STOPPING_PATIENCE = 5 # Stop after 5 epochs with no improvement
LR_SCHEDULER_PATIENCE = 2 # Reduce LR after 2 epochs with no improvement

# Set seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed(SEED)

print("Hyperparameters set.")
print(f"BATCH_SIZE: {BATCH_SIZE} (Adjust if OOM errors occur)")
print(f"LEARNING_RATE: {LEARNING_RATE}")
print(f"EPOCHS (Max): {EPOCHS}")


## ==============================================================================

# 4. Enhanced Data Augmentation ========================================
# Increased augmentation strength slightly
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)), # Crop a bit more aggressively
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Added ColorJitter
    transforms.RandomRotation(15), # Added RandomRotation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet stats
])
print("Transforms defined.")


## ==========================================================================

# 5. Data Loading and Splitting ======================================
# --- Simplified Data Loading ---
# Assume DATA_DIR contains class folders directly. Load everything first.
print(f"Loading dataset from: {DATA_DIR}")

# Create two instances of the dataset with different transforms
print("Creating dataset instances with respective transforms...")
# Use try-except block for robustness against missing data or corrupt images
try:
    dataset_for_train_split = datasets.ImageFolder(DATA_DIR, transform=train_transform)
    dataset_for_val_split = datasets.ImageFolder(DATA_DIR, transform=val_transform)
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Please ensure DATA_DIR points to the correct directory containing class folders.")
    raise # Re-raise the exception to stop execution

# Ensure classes are the same in both instances (sanity check)
assert dataset_for_train_split.classes == dataset_for_val_split.classes, "Class mismatch between dataset instances!"
num_classes = len(dataset_for_train_split.classes)
class_names = dataset_for_train_split.classes
class_to_idx = dataset_for_train_split.class_to_idx

# Handle case where dataset might be smaller than expected (important for splits)
if len(dataset_for_train_split) < 10: # Arbitrary small number check
     print(f"Warning: Dataset size ({len(dataset_for_train_split)}) is very small!")
     # Adjust split logic or handle as needed

print(f"Found {len(dataset_for_train_split)} images in {num_classes} classes.")
print(f"First few classes: {class_names[:10]}...") # Print only first few if many classes

# Calculate split sizes
dataset_size = len(dataset_for_train_split)
val_size = int(VALIDATION_SPLIT * dataset_size)
train_size = dataset_size - val_size

# --- Ensure validation set is not empty ---
if val_size == 0:
    raise ValueError(f"Validation split calculation resulted in 0 samples. Dataset size {dataset_size} might be too small for split {VALIDATION_SPLIT}")


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
    pin_memory=True if DEVICE.type == 'cuda' else False, # Pin memory only useful for GPU
    # drop_last=True # Consider if last batch size mismatch causes issues
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE, # Use same batch size or potentially larger for validation if memory allows
    shuffle=False, # No need to shuffle validation data
    num_workers=NUM_WORKERS,
    pin_memory=True if DEVICE.type == 'cuda' else False,
)

# Class distribution analysis (on the training set)
print("Analyzing class distribution in the training set...")
train_class_counts = torch.zeros(num_classes)
# Iterate through the train_dataset Subset to get labels more safely
train_targets = []
for idx in train_dataset.indices:
    target = dataset_for_train_split.targets[idx]
    train_targets.append(target)

unique_targets, counts = torch.unique(torch.tensor(train_targets), return_counts=True)
# Ensure counts tensor is correctly indexed by class index
for target_idx, count in zip(unique_targets, counts):
    train_class_counts[target_idx] = count


print("Training Set Class distribution (Top/Bottom 10 if > 20 classes):")
num_to_show = 10
if num_classes > 20:
    # Get sorted counts and indices
    sorted_counts, sorted_indices = torch.sort(train_class_counts, descending=True)
    top_indices = sorted_indices[:num_to_show]
    bottom_indices = sorted_indices[-num_to_show:]

    # Combine top and bottom, remove duplicates if any overlap
    combined_indices = torch.cat((top_indices, bottom_indices)).unique()
    plot_indices = combined_indices.numpy()
    plot_names = [class_names[i] for i in plot_indices]
    plot_counts = train_class_counts[plot_indices].numpy()
    title = "Training Set Class Distribution (Top/Bottom 10)"

else:
    plot_names = class_names
    plot_counts = train_class_counts.numpy()
    title = "Training Set Class Distribution"


plt.figure(figsize=(12, 6)) # Adjusted figure size
plt.bar(plot_names, plot_counts)
plt.xticks(rotation=65, ha='right', fontsize=8) # Rotate more, smaller font
plt.title(title)
plt.ylabel("Number of Samples")
plt.tight_layout() # Adjust layout
plt.savefig(LOG_DIR / 'class_distribution.png')
plt.show()
print("Data loading and splitting complete.")

##===========================================================================

# 6. Enhanced Model Setup ==============================================
print("Setting up the model (ResNet-50)...")
# --- Using ResNet-50 with updated weights API ---
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2) # <<< CHANGED TO RESNET50 V2 Weights

# Replace final layer (num_classes is derived above)
# The input features (2048 for ResNet50) are correctly inferred by model.fc.in_features
num_ftrs = model.fc.in_features
print(f"ResNet-50 original FC input features: {num_ftrs}")

model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 512),       # Project down from 2048 to 512
    nn.BatchNorm1d(512),            # BatchNorm for stability
    nn.ReLU(),
    nn.Dropout(0.5),                # Dropout for regularization
    nn.Linear(512, num_classes)     # Final layer to output class scores
)

# Freeze early layers (fine-tune layer 3, layer 4 and fc)
# This strategy is reasonable, but could be adjusted (e.g., fine-tune only layer4+fc)
print("Freezing initial layers, unfreezing layer3, layer4 and fc...")
for name, param in model.named_parameters():
    param.requires_grad = False

# Unfreeze layer 3
for param in model.layer3.parameters():
    param.requires_grad = True
# Unfreeze layer 4
for param in model.layer4.parameters():
    param.requires_grad = True
# Unfreeze the fully connected layer (already requires_grad by default after replacement, but good practice)
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)
print("Model setup complete (ResNet-50).")

# --- Optional: Print number of trainable parameters ---
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}") # Should be more than ResNet18

## =========================================================================

# 7. Enhanced Training Setup ===========================================
print("Setting up loss, optimizer, and scheduler...")
# Consider adding label smoothing if needed, especially with many classes
# criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
criterion = nn.CrossEntropyLoss()

# Filter parameters that require gradients for the optimizer
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                             lr=LEARNING_RATE,
                             weight_decay=WEIGHT_DECAY)

# Scheduler monitors validation accuracy and reduces LR if it plateaus
scheduler = ReduceLROnPlateau(optimizer,
                             mode='max', # Reduce LR when the metric stops increasing
                             factor=0.5, # New LR = LR * factor
                             patience=LR_SCHEDULER_PATIENCE,
                             verbose=True,
                             min_lr=1e-7) # Prevent LR from becoming too small

# Tracking variables
best_accuracy = 0.0
epochs_no_improve = 0 # Counter for early stopping
train_losses = []
val_accuracies = []
val_losses = []
learning_rates = [] # Track learning rate

print("Training setup complete.")

## =========================================================================

# 8. Training Loop with Early Stopping =================================
print(f"Starting training loop for up to {EPOCHS} epochs...")
print(f"Early stopping patience: {EARLY_STOPPING_PATIENCE} epochs")
print(f"LR scheduler patience: {LR_SCHEDULER_PATIENCE} epochs")
print("---")

for epoch in range(EPOCHS):
    # --- Training Phase ---
    model.train()
    running_loss = 0.0
    # Use train_size which is calculated correctly now
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]', unit='batch', leave=False)

    for images, labels in progress_bar:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping (optional but can prevent exploding gradients)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item() * images.size(0)

        # Update progress bar
        progress_bar.set_postfix(loss=f'{loss.item():.4f}', lr=f'{optimizer.param_groups[0]["lr"]:.1e}')

    # Use train_size calculated earlier
    epoch_train_loss = running_loss / train_size
    train_losses.append(epoch_train_loss)
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
    progress_bar.close() # Close the training progress bar

    # --- Validation Phase ---
    model.eval()
    val_running_loss = 0.0
    correct_preds = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Val]', unit='batch', leave=False)
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
            # Display running accuracy in validation progress bar
            val_progress_bar.set_postfix(acc=f'{(100 * correct_preds / total_samples):.2f}%')

    val_progress_bar.close() # Close the validation progress bar

    # Use val_size calculated earlier
    epoch_val_loss = val_running_loss / val_size
    epoch_val_accuracy = 100 * correct_preds / total_samples
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_accuracy)

    # End of Epoch Summary
    print(f'\nEpoch {epoch+1}/{EPOCHS} Summary:')
    print(f'Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_accuracy:.2f}% | LR: {current_lr:.1e}')

    # Learning rate scheduling step (based on validation accuracy)
    scheduler.step(epoch_val_accuracy)

    # --- Save Best Model ---
    if epoch_val_accuracy > best_accuracy:
        print(f'---> Validation accuracy improved ({best_accuracy:.2f}% -> {epoch_val_accuracy:.2f}%). Saving model...')
        best_accuracy = epoch_val_accuracy
        epochs_no_improve = 0 # Reset counter
        # Save necessary info for resuming or inference
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), # Save scheduler state too
            'best_accuracy': best_accuracy,
            'class_to_idx': class_to_idx, # Save the mapping
            'num_classes': num_classes, # Save num classes
            'class_names': class_names, # Save class names
        }, MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")
    else:
        epochs_no_improve += 1
        print(f'Validation accuracy did not improve for {epochs_no_improve} epoch(s). Best accuracy: {best_accuracy:.2f}%')

    # --- Early Stopping Check ---
    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
        print(f'\nEarly stopping triggered after {epoch + 1} epochs due to no improvement in validation accuracy for {EARLY_STOPPING_PATIENCE} consecutive epochs.')
        break # Exit the training loop

    print("---") # Separator between epoch logs


print(f"\nTraining finished after {epoch+1} epochs.") # Use final epoch value

## =========================================================================

# 9. Final Evaluation =================================================
print("\nLoading best model for final evaluation...")
# Load the best model state saved during training
# Use try-except for safer loading
try:
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE) # Load to current device

    # Re-create model structure (important if loading elsewhere)
    loaded_num_classes = checkpoint.get('num_classes', num_classes) # Get num_classes from checkpoint if saved
    print(f"Loading model for {loaded_num_classes} classes.")
    model = models.resnet50() # Create base ResNet50
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential( # Rebuild the same classifier head
         nn.Linear(num_ftrs, 512),
         nn.BatchNorm1d(512),
         nn.ReLU(),
         nn.Dropout(0.5),
         nn.Linear(512, loaded_num_classes)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE) # Move model to device
    model.eval() # Ensure model is in eval mode

    # Load class names from checkpoint if available, otherwise use the ones from data loading
    loaded_class_names = checkpoint.get('class_names', class_names)
    if len(loaded_class_names) != loaded_num_classes:
        print("Warning: Mismatch between loaded class names and number of classes. Using class names derived during data loading.")
        loaded_class_names = class_names # Fallback

    print(f"Best model from epoch {checkpoint['epoch']} with accuracy {checkpoint['best_accuracy']:.2f}% loaded.")

except FileNotFoundError:
    print(f"Error: Model checkpoint file not found at {MODEL_SAVE_PATH}. Cannot perform final evaluation.")
    print("Make sure training ran successfully and saved the model.")
    # Optionally, evaluate the model state at the end of training if no checkpoint exists
    # model.eval() # Ensure model is in eval mode
    # loaded_class_names = class_names # Use current class names
except Exception as e:
    print(f"Error loading checkpoint: {e}")
    # Optionally, proceed with the current model state if loading fails
    # model.eval()
    # loaded_class_names = class_names


# --- Re-run validation loop on the *loaded best model* to get metrics ---
# (or use the `all_labels` and `all_preds` from the epoch where the best model was saved,
# but rerunning ensures evaluation is on the truly *saved* best state)
print("\nRe-evaluating the best model on the validation set...")
final_val_running_loss = 0.0
final_correct_preds = 0
final_total_samples = 0
final_all_preds = []
final_all_labels = []

final_val_progress_bar = tqdm(val_loader, desc='Final Validation', unit='batch', leave=False)
with torch.no_grad():
    for images, labels in final_val_progress_bar:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, labels) # Can still calculate loss if needed
        final_val_running_loss += loss.item() * images.size(0)

        _, predicted = torch.max(outputs.data, 1)
        final_total_samples += labels.size(0)
        final_correct_preds += (predicted == labels).sum().item()

        final_all_preds.extend(predicted.cpu().numpy())
        final_all_labels.extend(labels.cpu().numpy())
        final_val_progress_bar.set_postfix(acc=f'{(100 * final_correct_preds / final_total_samples):.2f}%')

final_val_progress_bar.close()

if final_total_samples > 0:
    final_val_accuracy = 100 * final_correct_preds / final_total_samples
    final_val_loss = final_val_running_loss / final_total_samples
    print(f"Final Validation Results: Loss={final_val_loss:.4f}, Accuracy={final_val_accuracy:.2f}%")


    print("\nClassification Report (on validation set using best model):")
    # Ensure class_names aligns with the labels used in the report
    try:
        # Handle potential UndefinedMetricWarning for classes with no predicted samples
        report = classification_report(final_all_labels, final_all_preds, target_names=loaded_class_names, digits=4, zero_division=0)
        print(report)
        # Save classification report to file
        with open(LOG_DIR / 'classification_report.txt', 'w') as f:
             f.write(report)
        print(f"Classification report saved to {LOG_DIR / 'classification_report.txt'}")
    except ValueError as e:
        print(f"Error generating classification report: {e}")
        print("This might happen if the number of classes in `loaded_class_names` doesn't match the predicted labels.")


    # Plot Confusion Matrix
    print("\nPlotting Confusion Matrix...")
    try:
        cm = confusion_matrix(final_all_labels, final_all_preds)
        plt.figure(figsize=(max(12, loaded_num_classes // 3), max(10, loaded_num_classes // 4))) # Adjust size dynamically
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=loaded_class_names, yticklabels=loaded_class_names) # Turn off annotation if too many classes
        # Optionally add annotations only if matrix isn't too large
        if loaded_num_classes <= 50: # Example threshold
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=loaded_class_names, yticklabels=loaded_class_names, annot_kws={"size": 8})


        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix (Validation Set - Best Model)')
        plt.xticks(rotation=60, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(LOG_DIR / 'confusion_matrix.png')
        plt.show()
        print(f"Confusion matrix saved to {LOG_DIR / 'confusion_matrix.png'}")
    except ValueError as e:
         print(f"Error plotting confusion matrix: {e}")


# Plot training history (Loss, Accuracy, Learning Rate)
print("\nPlotting Training History...")
num_epochs_trained = len(train_losses) # Get actual number of epochs run
epochs_range = range(1, num_epochs_trained + 1)

plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.title('Validation Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.axhline(best_accuracy, color='r', linestyle='--', label=f'Best Acc: {best_accuracy:.2f}%') # Mark best accuracy
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(epochs_range, learning_rates, label='Learning Rate', color='g')
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.yscale('log') # Often better to view LR on log scale
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.savefig(LOG_DIR / 'training_history.png')
plt.show()
print(f"Training history plot saved to {LOG_DIR / 'training_history.png'}")


print(f"\n✅ Best validation accuracy achieved during training: {best_accuracy:.2f}%")
print(f"Best model saved to: {MODEL_SAVE_PATH}")
print(f"Logs and plots saved to: {LOG_DIR}")
print("\nScript finished.")