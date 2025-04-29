import torch
from PIL import Image
from load_model import load_model
from utils import preprocess_image

# Path to your trained model
MODEL_PATH = r"C:\Users\DELL\Documents\my_flask_app\models\resnet50_model.pth"
# MODEL_PATH = r"C:\Users\DELL\Documents\my_flask_app\models\test_modelv1.pth"
# MODEL_PATH = r"C:\Users\DELL\Documents\my_flask_app\models\test_modelv2.pth"

# Load model and class mappings
model, class_to_idx = load_model(MODEL_PATH, num_classes=None)
idx_to_class = {v: k for k, v in class_to_idx.items()}


def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)  # Get probabilities
        confidence, predicted_idx = torch.max(probabilities, dim=1)  # Get highest prob
        
        predicted_class = idx_to_class[predicted_idx.item()]
        confidence_percent = confidence.item() * 100  # Convert to 0-100%

        return {
            "class": predicted_class,
            "confidence": f"{round(confidence_percent, 2)}%"  # Append % symbol
        }


# def predict(image_path):
#     image = Image.open(image_path).convert("RGB")
#     input_tensor = preprocess_image(image)
#     with torch.no_grad():
#         output = model(input_tensor)
#         predicted_idx = torch.argmax(output, dim=1).item()
#         return idx_to_class[predicted_idx]
