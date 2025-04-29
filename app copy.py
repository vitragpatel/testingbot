# from flask import Flask, request, jsonify, send_from_directory, abort
# from uuid import uuid4
# from pathlib import Path
# from PIL import Image
# import logging
# from logging.handlers import RotatingFileHandler
# import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image, ImageFilter
from pathlib import Path
import logging
from logging.handlers import RotatingFileHandler
import os, io
import torch
import torchvision.transforms as T
import numpy as np

 
ALLOWED_EXT = {"png", "jpg", "jpeg", "webp"}
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("uploads/processed")
MODEL_CACHE_PATH = Path("model_cache.pth")
LOG_FILE = "app.log"
UPLOAD_FOLDER = 'static/uploads'

# Set up directories
for d in (UPLOAD_DIR, PROCESSED_DIR):
    d.mkdir(exist_ok=True, parents=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=1000000, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def allowed(fname):
    return "." in fname and fname.rsplit(".", 1)[1].lower() in ALLOWED_EXT



app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



from pathlib import Path
import torch, torchvision
from torchvision.models import resnet50, EfficientNet_B0_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torcheval.metrics.functional import multiclass_accuracy
from tqdm import tqdm

DATA_DIR      = Path(r"C:\Users\DELL\Documents\my_flask_app\mo_db")   # train/ and val/ subfolders
CHECKPOINTS   = Path("checkpoints"); CHECKPOINTS.mkdir(exist_ok=True)
NUM_CLASSES   = 50
BATCH_SIZE    = 64
EPOCHS        = 10
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- transforms ----------
train_tf = A.Compose([
    A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(0.2, 0.2, 0.2, 0.1),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_tf = A.Compose([
    A.Resize(256,256),
    A.CenterCrop(224,224),
    A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
    ToTensorV2(),
])

def wrap_ds(root, tf):  # ImageFolder + Albumentations
    base = ImageFolder(root)
    return [(tf(image=np.array(img))["image"], label) for img, label in base]

train_ds = wrap_ds(DATA_DIR/"train", train_tf)
val_ds   = wrap_ds(DATA_DIR/"val",   val_tf)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# ---------- model ----------
model = resnet50(weights="IMAGENET1K_V2")
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

opt = torch.optim.AdamW(model.parameters(),
                        lr=3e-4,      # ASCII hyphen
                        weight_decay=1e-4)

sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

# ---------- training loop ----------
for epoch in range(EPOCHS):
    model.train()
    for xb, yb in tqdm(train_dl, desc=f"epoch {epoch+1}/{EPOCHS}"):
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(xb), yb)
        loss.backward(); opt.step()
    sched.step()

    # ----- validation -----
    model.eval(); preds, gts = [], []
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(DEVICE)
            logits = model(xb)
            preds.append(logits.cpu()); gts.append(yb)
    acc = multiclass_accuracy(torch.cat(preds), torch.cat(gts), num_classes=NUM_CLASSES).item()
    print(f"val acc: {acc*100:.2f}%")

    torch.save(model.state_dict(), CHECKPOINTS/f"epoch{epoch+1:02d}_{acc*100:.1f}.pt")
model.load_state_dict(torch.load("checkpoints/epoch09_89.7.pt"))
torch.save(model, "model/deepfashion_resnet50.pt")   # <-- final file





# class_names = [  # 50 categories (shortened for readability)
#     "T‑shirt", "Shirt", "Blouse", "Top", "Sweater", "Cardigan", "Jacket",
#     "Coat", "Dress", "Jeans", "Shorts", "Skirt", "Pants", "Jumpsuit",
#     "Suit", "Hoodie", "Sweatpants", "Leggings", "Shoes", "Sandals",
#     # … continue to 50 …
# ]

# transform = T.Compose([
#     T.Resize(256),
#     T.CenterCrop(224),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]),
# ])





# # Home route
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Route to handle image upload and processing
# @app.route('/upload', methods=['POST'])
# def upload():
#     file = request.files.get("file")
#     if not file or file.filename == "":
#         return redirect(request.url)

#     # Save original
#     save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
#     file.save(save_path)

#     # Inference
#     image = Image.open(save_path).convert("RGB")
#     tensor = transform(image).unsqueeze(0).to(DEVICE)
#     with torch.no_grad():
#         probs = torch.softmax(model(tensor), dim=1)
#     top_prob, top_idx = probs.squeeze().max(0)
#     label = class_names[top_idx]

#     return render_template("result.html",
#                            filename=file.filename,
#                            label=label,
#                            confidence=f"{top_prob.item()*100:.1f}%")

# # Route to display the processed image
# @app.route('/uploads/<filename>')
# def show_image(filename):
#     return render_template('show_image.html', filename=filename)


if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 5000))
        app.run(host="0.0.0.0", port=port, debug=False)
    except Exception as e:
        logger.critical(f"Failed to start server: {str(e)}")