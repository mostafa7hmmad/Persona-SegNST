import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import tempfile
import numpy as np
import os

# ===============================
# TransformerNet
# ===============================
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    def forward(self, x):
        return self.conv2d(self.reflection_pad(x))

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels, 3, 1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, 3, 1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
    def forward(self, x):
        return x + self.in2(self.conv2(F.relu(self.in1(self.conv1(x)))))

class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super().__init__()
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=self.upsample, mode="nearest")
        return self.conv2d(self.reflection_pad(x))

class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvLayer(3, 32, 9, 1)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, 3, 2)
        self.in2 = nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, 3, 2)
        self.in3 = nn.InstanceNorm2d(128, affine=True)
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        self.deconv1 = UpsampleConvLayer(128, 64, 3, 1, upsample=2)
        self.in4 = nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, 3, 1, upsample=2)
        self.in5 = nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, 9, 1)
    def forward(self, x):
        y = F.relu(self.in1(self.conv1(x)))
        y = F.relu(self.in2(self.conv2(y)))
        y = F.relu(self.in3(self.conv3(y)))
        for res in [self.res1, self.res2, self.res3, self.res4, self.res5]:
            y = res(y)
        y = F.relu(self.in4(self.deconv1(y)))
        y = F.relu(self.in5(self.deconv2(y)))
        return self.deconv3(y)

# ===============================
# Load models
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
style_models = {
    "Candy": "candy.pth",
    "Mosaic": "mosaic.pth",
    "Rain Princess": "rain_princess.pth",
    "Udnie": "udnie.pth"
}
loaded_styles = {}
for name, path in style_models.items():
    model = TransformerNet().to(device)
    state_dict = torch.load(path, map_location=device)
    for k in list(state_dict.keys()):
        if k.endswith(("running_mean", "running_var")):
            del state_dict[k]
    model.load_state_dict(state_dict)
    model.eval()
    loaded_styles[name] = model

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mul(255))
])

def stylize_frame(frame, model):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image).cpu()
    output = output.squeeze().clamp(0, 255).numpy()
    output = output.transpose(1, 2, 0).astype("uint8")
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

# ===============================
# Streamlit UI
# ===============================
st.title("🎨 Style Transfer App")

uploaded_file = st.file_uploader("Upload an Image or Video", type=["jpg", "png", "jpeg", "mp4", "avi", "mov"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split("/")[0]

    # ===============================
    # For Images
    # ===============================
    if file_type == "image":
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        style_name = st.selectbox("Choose a style model", list(style_models.keys()))
        if st.button("Apply Style to Image"):
            model = loaded_styles[style_name]
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            styled = stylize_frame(frame, model)
            styled_rgb = cv2.cvtColor(styled, cv2.COLOR_BGR2RGB)
            st.image(styled_rgb, caption=f"Styled Image ({style_name})", use_column_width=True)

            # Save and allow download
            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            Image.fromarray(styled_rgb).save(out_path)
            with open(out_path, "rb") as f:
                st.download_button("Download Styled Image", f, file_name="styled_image.png")

    # ===============================
    # For Videos
    # ===============================
    elif file_type == "video":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.video(tfile.name)

        style_name = st.selectbox("Choose a style model", list(style_models.keys()))
        if st.button("Apply Style to Video"):
            model = loaded_styles[style_name]
            cap = cv2.VideoCapture(tfile.name)

            out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress = st.progress(0)

            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                styled = stylize_frame(frame, model)
                out.write(styled)
                count += 1
                progress.progress(min(count / total_frames, 1.0))

            cap.release()
            out.release()

            st.success("✅ Style transfer complete!")
            with open(out_path, "rb") as f:
                st.download_button("Download Styled Video", f, file_name="styled_video.mp4")
