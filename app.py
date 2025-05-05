from flask import Flask, request, jsonify
import os
import csv
from predict import predict
from werkzeug.utils import secure_filename

app = Flask(__name__)


UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created upload folder: {UPLOAD_FOLDER}")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Folder to temporarily save uploaded images
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return jsonify({"message": "Image classification API is up and running!"})


def save_predictions_to_csv(predictions, csv_path):
    # If file doesn't exist, write headers
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode="a", newline="") as csvfile:
        fieldnames = ["filename", "predicted_class", "confidence"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        # Add a separator for new responses
        writer.writerow({
            "filename": "-------------------",
            "predicted_class": "",
            "confidence": ""
        })

        for item in predictions:
            if "error" not in item:
                writer.writerow({
                    "filename": item["filename"],
                    "predicted_class": item["prediction"]["class"],
                    "confidence": item["prediction"]["confidence"]
                })
            else:
                # If error occurred, log it with empty class/confidence
                writer.writerow({
                    "filename": item["filename"],
                    "predicted_class": "ERROR: " + item["error"],
                    "confidence": ""
                })


@app.route("/predict", methods=["POST"])
def predict_images():
    if "images" not in request.files:
        return jsonify({"error": "No images uploaded."}), 400

    files = request.files.getlist("images")

    if not files:
        return jsonify({"error": "No files selected."}), 400

    predictions = []

    for image_file in files:
        if image_file.filename == "":
            continue  # Skip empty files

        filename = secure_filename(image_file.filename)
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        image_file.save(image_path)

        try:
            prediction = predict(image_path)
            predictions.append({
                "filename": filename,
                "prediction": prediction
            })
        except Exception as e:
            predictions.append({
                "filename": filename,
                "error": str(e)
            })
        finally:
            os.remove(image_path)

    #### ✨ Save to CSV
    # csv_output_path = os.path.join("your_folder", "predictions_log.csv")
    # save_predictions_to_csv(predictions, csv_output_path)
    
    return jsonify({"predictions": predictions})





#### for single image =============================
# @app.route("/predict", methods=["POST"])
# def predict_image():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded."}), 400

#     image_file = request.files["image"]
#     if image_file.filename == "":
#         return jsonify({"error": "No file selected."}), 400

#     filename = secure_filename(image_file.filename)
#     image_path = os.path.join(UPLOAD_FOLDER, filename)
#     image_file.save(image_path)

#     try:
#         prediction = predict(image_path)
#         return jsonify({"prediction": prediction})
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#     finally:
#         os.remove(image_path)

if __name__ == "__main__":
    app.run(debug=True)
