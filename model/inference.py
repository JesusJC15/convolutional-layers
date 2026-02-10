import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import io
import os
import base64
import json

from train import SimpleCNN

def model_fn(model_dir):
    # Load classes if present
    classes_path = os.path.join(model_dir, "classes.txt")
    if os.path.exists(classes_path):
        with open(classes_path, "r") as f:
            classes = [line.strip() for line in f.readlines()]
        num_classes = len(classes)
    else:
        classes = None
        num_classes = 6

    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location="cpu"))
    model.eval()
    model.classes = classes
    return model

def input_fn(request_body, content_type):
    if content_type == "application/json":
        payload = json.loads(request_body)
        img_bytes = base64.b64decode(payload["b64"])
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    else:
        raise ValueError("Unsupported content type")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

def predict_fn(input_data, model):
    with torch.no_grad():
        outputs = model(input_data)
        probs = F.softmax(outputs, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        pred_label = model.classes[pred_idx] if model.classes else pred_idx
        return {"pred_idx": pred_idx, "pred_label": pred_label, "probs": probs.tolist()}

def output_fn(prediction, content_type):
    return json.dumps(prediction), "application/json"