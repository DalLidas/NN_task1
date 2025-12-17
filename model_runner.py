"""
Классификация одного изображения с помощью обученной модели FractalNet.

Использование:
python model_runner.py --image ./dogs/beagle.jpg --model-dir ./model --data ./input/train/images
"""

import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# импорт модели из model_trainer_v2.py
from model_trainer_v2 import FractalNet


# ======================================================
# Load metadata
# ======================================================
def load_metadata(model_dir: Path):
    meta_file = model_dir / "metadata.json"
    if not meta_file.exists():
        raise FileNotFoundError(f"metadata.json not found in {model_dir}")

    data = json.loads(meta_file.read_text())
    if isinstance(data, list):
        return data[-1]
    return data


# ======================================================
# Load image
# ======================================================
def load_image(image_path, img_size=128):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)  # (1, C, H, W)
    return tensor, img


# ======================================================
# MAIN
# ======================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Путь к изображению для классификации")
    parser.add_argument("--model-dir", type=str, default="./model", help="Папка, где лежат *.pth и metadata.json")
    parser.add_argument("--data", type=str, required=True, help="Путь к корневой папке датасета Stanford Dogs (./input/train/images)")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model_dir = Path(args.model_dir)
    metadata = load_metadata(model_dir)

    model_path = Path(metadata["model_id"] + ".pth")
    model_path = model_dir / f"{metadata['model_id']}.pth"
    print("Using model file:", model_path)

    # параметры модели
    num_classes = 120
    channels = metadata.get("channels", [32, 64, 96, 128])
    classifier_dim = metadata.get("classifier_dim", 256)
    C = metadata.get("C", 4)
    drop_path = metadata.get("drop_path", 0.15)

    # === Load model ===
    model = FractalNet(
        num_classes=num_classes,
        channels=channels,
        classifier_dim=classifier_dim,
        C=C,
        drop_path_prob=drop_path
    ).to(device)

    state = torch.load(model_path, map_location=device)

    # если это финальная модель: там только state_dict
    if isinstance(state, dict) and "model_state" in state:
        # это checkpoint (плохой случай, но поддержим)
        model.load_state_dict(state["model_state"])
    else:
        # обычный state_dict
        model.load_state_dict(state)

    model.eval()

    # === Load image ===
    img_tensor, img = load_image(args.image)
    img_tensor = img_tensor.to(device)

    # === Predict ===
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    top5_prob, top5_idx = torch.topk(probabilities, 5)

    # === Load class names ===
    data_root = Path(args.data)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_root}")

    classes = sorted([d.name for d in data_root.glob("*") if d.is_dir()])

    # === Output ===
    print("\n=== Prediction (top-1) ===")
    print(f"{classes[top5_idx[0]]}  —  {top5_prob[0].item():.4f}")

    print("\n=== Top-5 ===")
    for p, idx in zip(top5_prob, top5_idx):
        print(f"{classes[idx]}  —  {p.item():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
