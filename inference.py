# inference.py
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu")

    # get model weights (handles both key types)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    num_classes = ckpt.get("num_classes")
    class_to_idx = ckpt.get("class_to_idx")
    img_size = ckpt.get("img_size", 224)

    # build classes list for numeric-index mapping
    classes = None
    if "classes" in ckpt:
        classes = ckpt["classes"]
    elif class_to_idx is not None:
        inv = {v: k for k, v in class_to_idx.items()}
        classes = [inv[i] for i in sorted(inv.keys())]

    return state, num_classes, classes, img_size


def build_model(state_dict, num_classes):
    # infer num_classes if missing
    if num_classes is None:
        for key in state_dict:
            if key.endswith("fc.weight"):
                num_classes = state_dict[key].shape[0]
                break
        if num_classes is None:
            raise RuntimeError("num_classes missing and cannot infer from checkpoint")

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def predict(model, img_path, classes, img_size, topk=3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = Image.open(img_path).convert("RGB")
    transform = get_transform(img_size)

    x = transform(img).unsqueeze(0).to(device)
    model = model.to(device)

    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out[0], dim=0)

    top_probs, top_idxs = torch.topk(probs, k=min(topk, probs.shape[0]))

    results = []
    for prob, idx in zip(top_probs.cpu().numpy(), top_idxs.cpu().numpy()):
        label = classes[idx] if classes is not None else f"class_{idx}"
        results.append((label, float(prob)))

    return results


def main():
    parser = argparse.ArgumentParser(description="Animal Classifier Inference Script")
    parser.add_argument("--image", "-i", required=True, help="Path to input image")
    parser.add_argument("--ckpt", "-c", default="animal_classifier.ckpt", help="Path to checkpoint file")
    parser.add_argument("--topk", "-k", type=int, default=3, help="Number of top predictions to show")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"ERROR: Image not found: {img_path}")
        sys.exit(1)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print("Loading checkpoint...")
    state, num_classes, classes, img_size = load_checkpoint(ckpt_path)

    print("Building model...")
    model = build_model(state, num_classes)

    print("Running prediction...")
    results = predict(model, img_path, classes, img_size, topk=args.topk)

    print(f"\nPredictions for: {img_path.name}")
    for i, (label, prob) in enumerate(results, 1):
        print(f"{i}. {label} — {prob*100:.2f}%")

    print("")


if __name__ == "__main__":
    main()
