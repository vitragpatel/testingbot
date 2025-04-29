from torchvision import transforms
from PIL import Image
import torch

# Define the exact transforms used for validation during training
# These should match the 'val_transform' from your training script
def get_preprocessing_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Function to apply transforms to a single PIL image
def preprocess_image(image: Image.Image):
    """
    Applies the necessary preprocessing transformations to a PIL Image
    and adds the batch dimension.

    Args:
        image (PIL.Image.Image): Input image.

    Returns:
        torch.Tensor: Preprocessed image tensor with batch dimension ([1, C, H, W]).
    """
    transform = get_preprocessing_transform()
    # Apply transform and add batch dimension (unsqueeze)
    tensor = transform(image).unsqueeze(0)
    return tensor


# from torchvision import transforms
# from PIL import Image

# # This should match what you used during training
# def preprocess_image(image: Image.Image):
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225]),
#     ])
#     return transform(image).unsqueeze(0)  # Add batch dimension
