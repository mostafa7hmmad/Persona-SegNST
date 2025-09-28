import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

def stylize(frame, model, device):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image).cpu()
    output = output.squeeze().clamp(0, 255).numpy()
    output = output.transpose(1, 2, 0).astype("uint8")
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return cv2.resize(output, (frame.shape[1], frame.shape[0]))
