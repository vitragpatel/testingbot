import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image # For loading images
import json # Potentially useful if class names are very long or contain special chars

# --- Configuration ---
UPLOAD_FOLDER = 'uploads' # Make sure this folder exists
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'} # Define allowed image types

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Get the directory where app.py is located
MODEL_FOLDER = 'models'
MODEL_FILENAME = 'resnet50_model.pth'
# Construct the path reliably
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FOLDER, MODEL_FILENAME)
# Or if 'models' is relative to where you run the script:
# MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_FILENAME)

print(f"Constructed model path: {MODEL_PATH}") # Add for debugging

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Model Loading and Setup (Done ONCE at Startup) ---

# 1. Define Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# 2. Load Model Structure and Weights
def load_trained_model(model_path, device):
    """Loads the saved model checkpoint."""
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Checkpoint loaded successfully from {model_path}")

        # --- Recreate the exact model architecture ---
        # Get number of classes from checkpoint
        num_classes = checkpoint.get('num_classes')
        if num_classes is None:
            # Fallback: Try getting from class_to_idx or class_names if num_classes wasn't saved explicitly
            if 'class_to_idx' in checkpoint:
                num_classes = len(checkpoint['class_to_idx'])
            elif 'class_names' in checkpoint:
                 num_classes = len(checkpoint['class_names'])
            else:
                 raise KeyError("Could not determine number of classes from checkpoint. Please save 'num_classes', 'class_to_idx', or 'class_names'.")

        print(f"Loading ResNet-50 model for {num_classes} classes.")
        model = models.resnet50() # Load base ResNet-50 structure

        # --- Rebuild the classifier head EXACTLY as during training ---
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

        # Load the learned weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval() # <<<=== IMPORTANT: Set model to evaluation mode!

        # --- Load Class Mappings ---
        if 'class_to_idx' in checkpoint:
            class_to_idx = checkpoint['class_to_idx']
            # Create inverse mapping (index to class name)
            idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        elif 'class_names' in checkpoint and 'idx_to_class' not in checkpoint:
             # If only class_names list was saved, assume standard 0-based indexing
             class_names = checkpoint['class_names']
             idx_to_class = {i: name for i, name in enumerate(class_names)}
        elif 'idx_to_class' in checkpoint:
            # If idx_to_class was saved directly
             idx_to_class = checkpoint['idx_to_class']
             # Ensure keys are integers if saved from json load sometimes
             idx_to_class = {int(k): v for k,v in idx_to_class.items()}
        else:
            raise KeyError("Checkpoint must contain 'class_to_idx' or 'class_names' or 'idx_to_class' for mapping predictions.")


        print(f"Model and class mappings loaded. Found {len(idx_to_class)} classes.")
        return model, idx_to_class

    except FileNotFoundError:
        print(f"Error: Model checkpoint file not found at {model_path}")
        return None, None
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return None, None

# --- Load the model globally when the app starts ---
loaded_model, loaded_idx_to_class = load_trained_model(MODEL_PATH, DEVICE)
if loaded_model is None:
    print("CRITICAL ERROR: Model could not be loaded. Predictions will fail.")
    # You might want to exit the app here depending on requirements
    # import sys
    # sys.exit(1)


# 3. Define Image Transformations (MUST match validation transforms)
#    Normalization stats are standard for ImageNet pre-trained models
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Prediction Function ---
def predict(image_path, model, transform, idx_to_class_map, device):
    """Predicts the class for a single image."""
    if model is None or idx_to_class_map is None:
        raise RuntimeError("Model or class mapping not loaded properly.")

    try:
        img = Image.open(image_path).convert('RGB') # Ensure image is RGB
        img_tensor = transform(img)
        # Add batch dimension (model expects batch_size x channels x height x width)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(device)

        with torch.no_grad(): # Disable gradient calculations for inference
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1) # Get probabilities
            confidence, predicted_idx = torch.max(probabilities, 1) # Get highest probability and its index

        predicted_class_idx = predicted_idx.item()
        predicted_class_name = idx_to_class_map.get(predicted_class_idx, "Unknown Class Index")
        confidence_score = confidence.item()

        # Return both class name and confidence
        return predicted_class_name, confidence_score

    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
        raise # Re-raise the exception to be caught by the API route


# --- API Routes ---
@app.route('/')
def index():
    return "Image Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict_images():
    if loaded_model is None:
         return jsonify({"error": "Model is not loaded. Cannot process requests."}), 500

    if "images" not in request.files:
        return jsonify({"error": "No 'images' key found in the request files."}), 400

    files = request.files.getlist("images")

    if not files or all(f.filename == '' for f in files):
        return jsonify({"error": "No image files selected or provided."}), 400

    predictions = []

    for image_file in files:
        if image_file and allowed_file(image_file.filename):
            filename = secure_filename(image_file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            try:
                image_file.save(image_path)
                print(f"Processing image: {filename}")

                # Call the prediction function
                predicted_class, confidence = predict(
                    image_path,
                    loaded_model,
                    inference_transform,
                    loaded_idx_to_class,
                    DEVICE
                )

                predictions.append({
                    "filename": filename,
                    "predicted_class": predicted_class,
                    "confidence": f"{confidence:.4f}" # Format confidence
                })
                print(f"Prediction for {filename}: {predicted_class} (Confidence: {confidence:.4f})")

            except RuntimeError as e: # Catch errors from predict function (e.g., model load issue)
                 print(f"Runtime error processing {filename}: {e}")
                 predictions.append({"filename": filename, "error": str(e)})
            except Exception as e:
                print(f"General error processing {filename}: {e}")
                predictions.append({"filename": filename, "error": f"Failed to process image: {str(e)}"})
            finally:
                # Clean up the saved image file
                if os.path.exists(image_path):
                    os.remove(image_path)
        elif image_file and image_file.filename != '':
             predictions.append({
                "filename": secure_filename(image_file.filename),
                "error": "File type not allowed."
             })


    # Optional: Save predictions to CSV (implement save_predictions_to_csv if needed)
    # csv_output_path = os.path.join("your_folder", "predictions_log.csv")
    # save_predictions_to_csv(predictions, csv_output_path)

    if not predictions:
         return jsonify({"error": "No valid image files processed."}), 400

    return jsonify({"predictions": predictions})

# --- Run the App ---
if __name__ == '__main__':
    # Use debug=False for production/deployment
    # Use host='0.0.0.0' to make it accessible on your network (be careful with security)
    app.run(debug=True, host='0.0.0.0', port=5000)