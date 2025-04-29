from datasets import load_dataset
from pathlib import Path
from PIL import Image
import io, tqdm, random


# HF_PATH = r"C:\Users\DELL\Documents\my_flask_app\img"
HF_PATH = str(Path("img").resolve())  # or Path("C:/.../img").resolve()
print("HF_PATH:.................", HF_PATH)

# DEST_ROOT = Path(r"C:\Users\DELL\Documents\my_flask_app\mo_db")
DEST_ROOT = Path("mo_db")
print("DEST_ROOT:.................", DEST_ROOT)

VAL_FRACTION = 0.05

# Load from folder using Hugging Face Datasets
print("start............Loading dataset from folder...")
ds = load_dataset("imagefolder", data_dir=HF_PATH, split="train", streaming=True)
print("end............Loading dataset from folder...")
label_names = ds.features["label"].names

for i, row in enumerate(tqdm.tqdm(ds)):
    print(f"Processing {i}....................")
    cls = label_names[row["label"]]
    subset = "val" if random.random() < VAL_FRACTION else "train"
    dest_dir = DEST_ROOT / subset / cls
    dest_dir.mkdir(parents=True, exist_ok=True)

    fname = f'{i:07d}.jpg'
    img = row["image"]
    img.convert("RGB").save(dest_dir / fname, quality=95)

print("✅ Done! Images split into train/val folders at:", DEST_ROOT)
