import torch
from PIL import Image

from config import DEVICE, WEIGHTS_PATH, CLASSES
from model import IDCClassifier
from dataset import eval_transform


def predict_image(image_path, model=None, device=DEVICE):
    """
    Predict a single image OR a list of image paths.

    Parameters
    ----------
    image_path : str | list[str]  — single path or list of paths
    model      : IDCClassifier    — if None, loads from WEIGHTS_PATH

    Returns
    -------
    If single path  → (label: str, confidence: float)
    If list of paths → list of (label: str, confidence: float)
    """
    if model is None:
        model = IDCClassifier().to(device)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    # Handle batch input (list of paths)
    if isinstance(image_path, list):
        tensors = [eval_transform(Image.open(p).convert("RGB")) for p in image_path]
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            probs = model.predict_proba(batch).squeeze(1).cpu().tolist()
        return [(CLASSES[1] if p > 0.5 else CLASSES[0], p) for p in probs]

    # Handle single image
    image  = Image.open(image_path).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)  # (1, 3, 50, 50)

    with torch.no_grad():
        prob = model.predict_proba(tensor).item()

    label = CLASSES[1] if prob > 0.5 else CLASSES[0]
    return label, prob


def predict_batch(image_paths, model=None, device=DEVICE):
    """
    Predict a list of image paths efficiently in one forward pass.

    Parameters
    ----------
    image_paths : list[str]

    Returns
    -------
    results : list of (label: str, confidence: float)
    """
    if model is None:
        model = IDCClassifier().to(device)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()

    tensors = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        tensors.append(eval_transform(img))

    batch = torch.stack(tensors).to(device)  # (N, 3, 50, 50)

    with torch.no_grad():
        probs = model.predict_proba(batch).squeeze(1).cpu().tolist()

    return [(CLASSES[1] if p > 0.5 else CLASSES[0], p) for p in probs]


if __name__ == "__main__":
    import os

    # Run predictions on all images in the data/ folder
    data_dir = "data"
    image_files = sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    if not image_files:
        print("No images found in data/ folder.")
    else:
        print(f"Running predictions on {len(image_files)} images...\n")
        results = predict_batch(image_files)
        for path, (label, conf) in zip(image_files, results):
            print(f"  {os.path.basename(path):20s}  →  {label}  (conf: {conf:.4f})")
