import torch
import torch.nn as nn
from torchvision import models

def load_model(model_path, num_classes=None):
    """Loads the trained ResNet18 model with the custom classifier head."""
    print(f"Loading model checkpoint from: {model_path}")
    # Load checkpoint onto CPU initially to avoid GPU issues if model was saved on GPU
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    # Load class mapping
    if 'class_to_idx' not in checkpoint:
        raise KeyError("Checkpoint must contain 'class_to_idx' mapping.")
    class_to_idx = checkpoint['class_to_idx']

    # Determine the number of classes
    if num_classes is None:
        num_classes = len(class_to_idx)
        print(f"Inferred number of classes: {num_classes}")
    elif num_classes != len(class_to_idx):
         print(f"Warning: Provided num_classes ({num_classes}) differs from checkpoint classes ({len(class_to_idx)}). Using checkpoint's class count.")
         num_classes = len(class_to_idx)


    # Instantiate the base model architecture (use weights=None as state_dict will overwrite)
    model = models.resnet18(weights=None) # Or use weights=models.ResNet18_Weights.IMAGENET1K_V1 if you prefer consistency

    # --- CRITICAL: Recreate the *exact* same fc layer structure as during training ---
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.BatchNorm1d(512), # Must match training!
        nn.ReLU(),
        nn.Dropout(0.5),    # Must match training!
        nn.Linear(512, num_classes) # Output layer
    )
    print("Model structure with custom fc layer created.")

    # Load the state dictionary
    if 'model_state_dict' not in checkpoint:
         raise KeyError("Checkpoint must contain 'model_state_dict'.")

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model state dictionary loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading state_dict: {e}")
        print("This often indicates a mismatch between the model architecture defined here and the one saved in the checkpoint.")
        raise e

    # Set the model to evaluation mode
    model.eval()
    print("Model set to evaluation mode.")

    return model, class_to_idx


# import torch
# import torch.nn as nn
# from torchvision import models

# def load_model(model_path, num_classes=None):
#     checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
#     class_to_idx = checkpoint['class_to_idx']
    
#     if num_classes is None:
#         num_classes = len(class_to_idx)

#     model = models.resnet18(weights=None)
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     model.load_state_dict(checkpoint['model_state_dict'])

#     model.eval()
#     return model, class_to_idx
