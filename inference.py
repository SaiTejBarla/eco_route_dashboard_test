import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

MODEL_PATH = "bin_classifier.pth"
CLASSES = ["Empty", "Half", "Full"]

def load_model():
    """Load trained MobileNetV2 model for inference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

def predict_image(img_path):
    """Run inference on a single bin image and return predicted class."""
    model, device = load_model()
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return CLASSES[predicted.item()]

if __name__ == "__main__":
    test_img = "sample_bin.jpg"  # Replace with your test image
    print("Prediction:", predict_image(test_img))
