import base64
import io
import os
import time
import types

import cv2
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tensorflow import keras
from torchvision import transforms

# ===============================
# Page Configuration (Must be first)
# ===============================
st.set_page_config(
    page_title="Persona-SegNST Studio",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===============================
# Patch for Keras 3 / segmentation_models compatibility
# ===============================
if not hasattr(keras.utils, "generic_utils"):
    keras.utils.generic_utils = types.SimpleNamespace()
    keras.utils.generic_utils.to_list = lambda x: x if isinstance(x, list) else [x]
    keras.utils.generic_utils.get_custom_objects = lambda: {}

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STYLE_CATALOG = {
    "Mosaic": {
        "file": "mosaic.pth",
        "tag": "Stained Glass",
        "desc": "Geometric leaded-glass facets with bold structural contours.",
        "gradient": "linear-gradient(135deg, #0ea5e9 0%, #6366f1 50%, #f59e0b 100%)",
        "swatches": ["#0ea5e9", "#6366f1", "#f59e0b"],
    },
    "Candy": {
        "file": "candy.pth",
        "tag": "Pop Cubism",
        "desc": "Vivid chromatic blocks with energetic painterly brush sweeps.",
        "gradient": "linear-gradient(135deg, #ec4899 0%, #f97316 55%, #facc15 100%)",
        "swatches": ["#ec4899", "#f97316", "#facc15"],
    },
    "Rain": {
        "file": "rain_princess.pth",
        "tag": "Impressionist",
        "desc": "Luminous rain-washed strokes with warm amber and cobalt tones.",
        "gradient": "linear-gradient(135deg, #1d4ed8 0%, #06b6d4 55%, #fbbf24 100%)",
        "swatches": ["#1d4ed8", "#06b6d4", "#fbbf24"],
    },
    "Udnie": {
        "file": "udnie.pth",
        "tag": "Abstract Modern",
        "desc": "High-contrast modernist abstractions with sculptural rhythm.",
        "gradient": "linear-gradient(135deg, #10b981 0%, #3b82f6 55%, #a855f7 100%)",
        "swatches": ["#10b981", "#3b82f6", "#a855f7"],
    },
}

MODE_CATALOG = {
    "StyleOnMask": {
        "label": "Stylize Subject",
        "short": "StyleOnMask",
        "desc": "Applies artistic style strictly to the segmented person while keeping the background photorealistic.",
    },
    "StyleOnBG": {
        "label": "Stylize Background",
        "short": "StyleOnBG",
        "desc": "Preserves the natural foreground person while transforming the surrounding scene into art.",
    },
    "Style": {
        "label": "Full-Frame Style",
        "short": "Style",
        "desc": "Transfers the selected neural style uniformly across the entire composition.",
    },
    "Segmentation": {
        "label": "Subject Isolation",
        "short": "Segmentation",
        "desc": "Extracts the human silhouette onto a customizable studio backdrop color.",
    },
}


# ===============================
# TransformerNet Architecture
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
# Cached Model Loading
# ===============================
@st.cache_resource(show_spinner="Initializing neural style checkpoints...")
def load_style_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = {}
    for name, meta in STYLE_CATALOG.items():
        path = os.path.join(BASE_DIR, meta["file"])
        model = TransformerNet().to(device)
        try:
            state_dict = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(path, map_location=device)
        for k in list(state_dict.keys()):
            if k.endswith(("running_mean", "running_var")):
                del state_dict[k]
        model.load_state_dict(state_dict)
        model.eval()
        loaded[name] = model
    return loaded, device


@st.cache_resource(show_spinner="Loading MobileNetV2-UNet segmentation model...")
def load_segmentation_model():
    model_path = os.path.join(BASE_DIR, "best_Unet.h5")
    seg_model = keras.models.load_model(model_path, compile=False)
    preprocess_fn = sm.get_preprocessing("mobilenetv2")

    @tf.function
    def predict_fn(input_tensor):
        return seg_model(input_tensor, training=False)

    return predict_fn, preprocess_fn


STYLE_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Lambda(lambda x: x.mul(255))]
)


# ===============================
# Inference & Compositing Helpers
# ===============================
def prepare_frame(raw_bgr: np.ndarray, preserve_aspect: bool, target_size: int) -> np.ndarray:
    h, w = raw_bgr.shape[:2]
    if not preserve_aspect:
        size = max(128, (target_size // 4) * 4)
        return cv2.resize(raw_bgr, (size, size), interpolation=cv2.INTER_AREA)

    scale = target_size / float(max(h, w))
    new_w = max(64, int(round(w * scale / 4.0)) * 4)
    new_h = max(64, int(round(h * scale / 4.0)) * 4)
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    return cv2.resize(raw_bgr, (new_w, new_h), interpolation=interp)


def run_segmentation(
    frame_bgr: np.ndarray,
    predict_fn,
    preprocess_fn,
    norm_mode: str,
) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)

    if norm_mode == "Training [0, 1]":
        inp = resized.astype(np.float32) / 255.0
    else:
        inp = preprocess_fn(resized).astype(np.float32)

    inp = np.expand_dims(inp, axis=0)
    prob_224 = predict_fn(tf.constant(inp))[0].numpy()
    if prob_224.ndim == 3:
        prob_224 = prob_224[:, :, 0]
    prob_full = cv2.resize(prob_224, (w, h), interpolation=cv2.INTER_LINEAR)
    prob_full = np.clip(prob_full, 0.0, 1.0)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return prob_full, elapsed_ms


def build_alpha_matte(
    prob_map: np.ndarray,
    threshold: float,
    feather_px: int,
    soft_matte: bool,
) -> np.ndarray:
    if soft_matte:
        # Smooth sigmoid-like ramp centered on threshold for natural edge transitions
        alpha_1ch = np.clip((prob_map - (threshold - 0.15)) / 0.30, 0.0, 1.0).astype(np.float32)
    else:
        alpha_1ch = (prob_map > threshold).astype(np.float32)

    if feather_px > 0:
        k = feather_px * 2 + 1
        alpha_1ch = cv2.GaussianBlur(alpha_1ch, (k, k), 0)
        alpha_1ch = np.clip(alpha_1ch, 0.0, 1.0)

    return np.repeat(alpha_1ch[:, :, np.newaxis], 3, axis=2)


def stylize_frame(frame_bgr: np.ndarray, model: nn.Module, device: torch.device) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    h, w = frame_bgr.shape[:2]
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tensor = STYLE_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor).cpu()
    output = output.squeeze(0).clamp(0, 255).numpy()
    output = output.transpose(1, 2, 0).astype(np.uint8)
    output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    if output_bgr.shape[:2] != (h, w):
        output_bgr = cv2.resize(output_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return output_bgr, elapsed_ms


def hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def composite_output(
    frame_bgr: np.ndarray,
    styled_bgr: np.ndarray | None,
    alpha_3ch: np.ndarray,
    mode: str,
    style_strength: float,
    bg_hex: str,
) -> np.ndarray:
    frame_f = frame_bgr.astype(np.float32)

    if styled_bgr is not None:
        blended_style = styled_bgr.astype(np.float32) * style_strength + frame_f * (1.0 - style_strength)
    else:
        blended_style = frame_f

    if mode == "Style":
        out = blended_style
    elif mode == "Segmentation":
        bg_bgr = hex_to_bgr(bg_hex)
        bg = np.full_like(frame_f, bg_bgr, dtype=np.float32)
        out = frame_f * alpha_3ch + bg * (1.0 - alpha_3ch)
    elif mode == "StyleOnMask":
        out = blended_style * alpha_3ch + frame_f * (1.0 - alpha_3ch)
    elif mode == "StyleOnBG":
        out = frame_f * alpha_3ch + blended_style * (1.0 - alpha_3ch)
    else:
        out = frame_f

    return np.clip(out, 0, 255).astype(np.uint8)


def encode_image_bytes(img_rgb: np.ndarray, fmt: str = "PNG", alpha_1ch: np.ndarray | None = None) -> bytes:
    buf = io.BytesIO()
    if fmt == "RGBA_PNG" and alpha_1ch is not None:
        alpha_u8 = np.clip(alpha_1ch * 255.0, 0, 255).astype(np.uint8)
        rgba = np.dstack([img_rgb, alpha_u8])
        Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    elif fmt == "JPEG":
        Image.fromarray(img_rgb, mode="RGB").save(buf, format="JPEG", quality=95)
    else:
        Image.fromarray(img_rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def b64_png(img_rgb: np.ndarray) -> str:
    return base64.b64encode(encode_image_bytes(img_rgb, "PNG")).decode("ascii")


# ===============================
# Custom Studio CSS & Interactive Split-Compare Component
# ===============================
def inject_studio_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        :root {
            --studio-bg: #080a10;
            --studio-surface: #111522;
            --studio-surface-2: #171d2f;
            --studio-border: rgba(148, 163, 184, 0.16);
            --studio-border-strong: rgba(124, 92, 255, 0.45);
            --studio-text: #f1f5f9;
            --studio-muted: #94a3b8;
            --studio-accent: #7c5cff;
            --studio-accent-2: #22d3ee;
            --studio-radius: 14px;
            color-scheme: only dark;
            accent-color: var(--studio-accent);
        }

        ::selection {
            background: rgba(124, 92, 255, 0.42);
            color: #ffffff;
        }

        html, body, [data-testid="stAppViewContainer"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--studio-bg);
            background-image:
                radial-gradient(900px 420px at 14% -8%, rgba(124, 92, 255, 0.16), transparent 60%),
                radial-gradient(720px 380px at 88% 0%, rgba(34, 211, 238, 0.12), transparent 58%);
            color: var(--studio-text);
            accent-color: var(--studio-accent);
            color-scheme: only dark;
            scrollbar-color: #64748b #0f1422;
            scrollbar-width: auto;
        }

        [data-testid="stHeader"] {
            background: rgba(8, 10, 16, 0.72);
            backdrop-filter: blur(12px);
            border-bottom: 1px solid var(--studio-border);
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .block-container {
            padding-top: 2.1rem;
            padding-bottom: 2.5rem;
            max-width: 1380px;
        }

        /* Sidebar Atelier Panel */
        [data-testid="stSidebar"] {
            background-color: #0d111c;
            border-right: 1px solid var(--studio-border);
            color: var(--studio-text);
            color-scheme: only dark;
            accent-color: var(--studio-accent);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
        }

        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            padding: 0.25rem 0 0.9rem 0;
            border-bottom: 1px solid var(--studio-border);
            margin-bottom: 1rem;
        }

        .sidebar-logo {
            width: 36px;
            height: 36px;
            border-radius: 10px;
            background: linear-gradient(135deg, #7c5cff, #22d3ee);
            display: grid;
            place-items: center;
            font-weight: 800;
            font-size: 1rem;
            color: #06080f;
            box-shadow: 0 6px 20px rgba(124, 92, 255, 0.35);
            flex-shrink: 0;
        }

        .sidebar-brand-title {
            font-size: 0.98rem;
            font-weight: 700;
            color: #f8fafc;
            letter-spacing: -0.01em;
            line-height: 1.2;
        }

        .sidebar-brand-sub {
            font-size: 0.74rem;
            color: var(--studio-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        .sidebar-section-label {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: #94a3b8;
            margin: 1rem 0 0.45rem 0;
        }

        /* Active Style Preview Card in Sidebar */
        .style-preview-card {
            background: var(--studio-surface);
            border: 1px solid var(--studio-border);
            border-radius: 12px;
            padding: 0.75rem 0.85rem;
            margin: 0.35rem 0 0.9rem 0;
        }

        .style-swatch-bar {
            height: 7px;
            border-radius: 999px;
            margin-bottom: 0.6rem;
        }

        .style-meta-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.25rem;
        }

        .style-meta-name {
            font-weight: 700;
            font-size: 0.88rem;
            color: #f8fafc;
        }

        .style-meta-tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            padding: 0.15rem 0.45rem;
            border-radius: 999px;
            background: rgba(124, 92, 255, 0.16);
            border: 1px solid rgba(124, 92, 255, 0.35);
            color: #c4b5fd;
        }

        .style-meta-desc {
            font-size: 0.77rem;
            color: var(--studio-muted);
            line-height: 1.4;
        }

        /* Hero Command Header */
        .studio-hero {
            background: linear-gradient(180deg, rgba(23, 29, 47, 0.82) 0%, rgba(17, 21, 34, 0.9) 100%);
            border: 1px solid var(--studio-border);
            border-radius: 18px;
            padding: 1.25rem 1.5rem;
            margin-bottom: 1.25rem;
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.32);
        }

        .studio-hero h1 {
            margin: 0;
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.025em;
            background: linear-gradient(95deg, #ffffff 0%, #c4b5fd 55%, #67e8f9 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .studio-hero p {
            margin: 0.25rem 0 0 0;
            color: var(--studio-muted);
            font-size: 0.88rem;
        }

        .telemetry-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .telemetry-pill {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.74rem;
            color: #cbd5e1;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid var(--studio-border);
            padding: 0.34rem 0.7rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
        }

        .telemetry-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #22d3ee;
            box-shadow: 0 0 8px rgba(34, 211, 238, 0.8);
        }

        .telemetry-dot.violet {
            background: #7c5cff;
            box-shadow: 0 0 8px rgba(124, 92, 255, 0.85);
        }

        /* Metric Strip Cards */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.8rem;
            margin: 0.2rem 0 1.15rem 0;
        }

        @media (max-width: 900px) {
            .metric-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        .metric-card {
            background: var(--studio-surface);
            border: 1px solid var(--studio-border);
            border-radius: 13px;
            padding: 0.8rem 1rem;
        }

        .metric-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            color: var(--studio-muted);
            margin-bottom: 0.25rem;
        }

        .metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.12rem;
            font-weight: 700;
            color: #f8fafc;
        }

        .metric-sub {
            font-size: 0.72rem;
            color: #64748b;
            margin-top: 0.12rem;
        }

        /* Mode Showcase Cards (Empty State) */
        .mode-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.9rem;
            margin-top: 1rem;
        }

        @media (max-width: 960px) {
            .mode-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        .mode-card {
            background: var(--studio-surface);
            border: 1px solid var(--studio-border);
            border-radius: 14px;
            padding: 1rem 1.05rem;
            transition: border-color 160ms ease, transform 160ms ease;
        }

        .mode-card:hover {
            border-color: var(--studio-border-strong);
            transform: translateY(-2px);
        }

        .mode-card-kicker {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.68rem;
            color: #67e8f9;
            margin-bottom: 0.3rem;
        }

        .mode-card-title {
            font-weight: 700;
            font-size: 0.95rem;
            color: #f8fafc;
            margin-bottom: 0.35rem;
        }

        .mode-card-desc {
            font-size: 0.8rem;
            color: var(--studio-muted);
            line-height: 1.45;
        }

        /* File Uploader Polish */
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(17, 21, 34, 0.88);
            border: 1.5px dashed rgba(124, 92, 255, 0.5);
            border-radius: 16px;
            padding: 1.2rem;
            transition: border-color 180ms ease, background-color 180ms ease;
        }

        [data-testid="stFileUploaderDropzone"]:hover {
            border-color: #22d3ee;
            background: rgba(23, 29, 47, 0.95);
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
            background: var(--studio-surface);
            padding: 0.32rem;
            border-radius: 12px;
            border: 1px solid var(--studio-border);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 9px;
            padding: 0.45rem 0.95rem;
            color: var(--studio-muted);
            font-weight: 600;
            font-size: 0.86rem;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(124, 92, 255, 0.28), rgba(34, 211, 238, 0.18));
            color: #ffffff !important;
            border: 1px solid rgba(124, 92, 255, 0.45);
        }

        /* Buttons */
        .stDownloadButton > button, .stButton > button {
            background: linear-gradient(135deg, #6d4aff 0%, #4f46e5 100%);
            color: #ffffff;
            border: 1px solid rgba(196, 181, 253, 0.35);
            border-radius: 10px;
            font-weight: 600;
            padding: 0.52rem 1rem;
            transition: transform 140ms ease, box-shadow 140ms ease, filter 140ms ease;
        }

        .stDownloadButton > button:hover, .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 22px rgba(124, 92, 255, 0.35);
            filter: brightness(1.08);
            color: #ffffff;
        }

        /* Image Frame Containers */
        .canvas-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.45rem;
            font-size: 0.78rem;
            font-weight: 600;
            color: #cbd5e1;
            font-family: 'JetBrains Mono', monospace;
        }

        .canvas-badge {
            font-size: 0.68rem;
            padding: 0.14rem 0.5rem;
            border-radius: 999px;
            background: rgba(34, 211, 238, 0.14);
            border: 1px solid rgba(34, 211, 238, 0.32);
            color: #67e8f9;
        }

        [data-testid="stImage"] img {
            border-radius: 14px;
            border: 1px solid var(--studio-border);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.38);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_split_compare(orig_rgb: np.ndarray, res_rgb: np.ndarray, mode_label: str, style_label: str):
    h, w = orig_rgb.shape[:2]
    orig_b64 = b64_png(orig_rgb)
    res_b64 = b64_png(res_rgb)

    aspect_pct = max(48, min(115, int(round((h / float(w)) * 100))))
    iframe_height = max(470, min(680, int(round(560 * (h / float(w)))) + 105))

    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <style>
        :root {{
          --bg: #0c101b;
          --surface: #111625;
          --border: rgba(148, 163, 184, 0.2);
          --text: #f1f5f9;
          --muted: #94a3b8;
          --accent: #7c5cff;
          --cyan: #22d3ee;
          color-scheme: only dark;
          accent-color: var(--accent);
        }}
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          padding: 0;
          background: #080a10;
          color: var(--text);
          font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif;
          color-scheme: only dark;
          accent-color: var(--accent);
        }}
        .shell {{
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 14px;
          max-width: 680px;
          margin: 0 auto;
          box-shadow: 0 16px 40px rgba(0, 0, 0, 0.45);
        }}
        .stage {{
          position: relative;
          width: 100%;
          aspect-ratio: {w} / {h};
          max-height: 540px;
          margin: 0 auto;
          overflow: hidden;
          border-radius: 12px;
          border: 1px solid rgba(148, 163, 184, 0.22);
          background: #07090f;
          --split: 50%;
        }}
        .stage img {{
          position: absolute;
          inset: 0;
          width: 100%;
          height: 100%;
          object-fit: contain;
          display: block;
          pointer-events: none;
        }}
        .layer-result {{
          clip-path: inset(0 0 0 var(--split));
        }}
        .divider {{
          position: absolute;
          top: 0;
          bottom: 0;
          left: var(--split);
          width: 2px;
          transform: translateX(-1px);
          background: linear-gradient(180deg, #22d3ee, #7c5cff);
          box-shadow: 0 0 14px rgba(124, 92, 255, 0.85);
          pointer-events: none;
        }}
        .handle {{
          position: absolute;
          top: 50%;
          left: var(--split);
          transform: translate(-50%, -50%);
          width: 38px;
          height: 38px;
          border-radius: 50%;
          background: rgba(15, 23, 42, 0.92);
          border: 2px solid #22d3ee;
          color: #f8fafc;
          display: grid;
          place-items: center;
          font-size: 13px;
          font-weight: 700;
          box-shadow: 0 6px 20px rgba(0, 0, 0, 0.55);
          pointer-events: none;
          user-select: none;
        }}
        .pill {{
          position: absolute;
          top: 12px;
          padding: 4px 10px;
          border-radius: 999px;
          font-size: 11px;
          font-weight: 700;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          background: rgba(8, 10, 16, 0.78);
          border: 1px solid rgba(148, 163, 184, 0.3);
          color: #f8fafc;
          backdrop-filter: blur(8px);
          pointer-events: none;
        }}
        .pill-left {{ left: 12px; }}
        .pill-right {{
          right: 12px;
          border-color: rgba(124, 92, 255, 0.6);
          color: #ddd6fe;
        }}
        .controls {{
          margin-top: 12px;
          display: flex;
          align-items: center;
          gap: 12px;
        }}
        .controls label {{
          font-size: 12px;
          font-weight: 600;
          color: var(--muted);
          white-space: nowrap;
        }}
        .readout {{
          font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
          font-size: 12px;
          font-weight: 600;
          color: #67e8f9;
          min-width: 40px;
          text-align: right;
        }}
        input[type="range"] {{
          --slider-track-size: 6px;
          --slider-thumb-size: 18px;
          --slider-track-radius: 999px;
          --slider-thumb-radius: 50%;
          --slider-track-color: #64748b;
          --slider-fill-color: #7c5cff;
          --slider-thumb-color: #22d3ee;
          --slider-direction: to right;
          --attr-min: attr(min type(<number>), 0);
          --attr-max: attr(max type(<number>), 100);
          --control-value: attr(value type(<number>), 50);
          --slider-progress: calc(progress(var(--control-value), var(--attr-min), var(--attr-max)) * 100%);
          appearance: none;
          inline-size: 100%;
          block-size: 36px;
          margin: 0;
          padding: 0;
          border: 0;
          background: transparent;
          cursor: pointer;
        }}
        @supports not (opacity: progress(1, 0, 100)) {{
          input[type="range"] {{
            --slider-progress: calc((var(--control-value) - var(--attr-min)) / (var(--attr-max) - var(--attr-min)) * 100%);
          }}
        }}
        input[type="range"]:focus-visible {{
          outline: 2px solid #22d3ee;
          outline-offset: 4px;
          border-radius: 6px;
        }}
        /* WebKit / Blink */
        input[type="range"]::-webkit-slider-runnable-track {{
          block-size: var(--slider-track-size);
          border: 0;
          border-radius: var(--slider-track-radius);
          background: linear-gradient(
            var(--slider-direction),
            var(--slider-fill-color) var(--slider-progress),
            var(--slider-track-color) var(--slider-progress)
          );
        }}
        input[type="range"]::-webkit-slider-thumb {{
          -webkit-appearance: none;
          appearance: none;
          inline-size: var(--slider-thumb-size);
          block-size: var(--slider-thumb-size);
          margin-block-start: calc((var(--slider-track-size) - var(--slider-thumb-size)) / 2);
          border: 2px solid #080a10;
          border-radius: var(--slider-thumb-radius);
          background: var(--slider-thumb-color);
          box-shadow: 0 0 10px rgba(34, 211, 238, 0.65);
        }}
        /* Firefox */
        input[type="range"]::-moz-range-track {{
          block-size: var(--slider-track-size);
          border: 0;
          border-radius: var(--slider-track-radius);
          background: var(--slider-track-color);
        }}
        input[type="range"]::-moz-range-progress {{
          block-size: var(--slider-track-size);
          border-radius: var(--slider-track-radius);
          background: var(--slider-fill-color);
        }}
        input[type="range"]::-moz-range-thumb {{
          inline-size: var(--slider-thumb-size);
          block-size: var(--slider-thumb-size);
          border: 2px solid #080a10;
          border-radius: var(--slider-thumb-radius);
          background: var(--slider-thumb-color);
        }}
      </style>
    </head>
    <body>
      <div class="shell">
        <div class="stage" id="stage">
          <img src="data:image/png;base64,{orig_b64}" alt="Original input image" />
          <img class="layer-result" src="data:image/png;base64,{res_b64}" alt="Processed result image" />
          <div class="divider"></div>
          <div class="handle" aria-hidden="true">&#8596;</div>
          <span class="pill pill-left">Original</span>
          <span class="pill pill-right">{mode_label} &middot; {style_label}</span>
        </div>
        <div class="controls">
          <label for="compare-slider">Split Wipe</label>
          <input
            type="range"
            id="compare-slider"
            min="0"
            max="100"
            step="1"
            value="50"
            aria-label="Compare original and stylized image"
            aria-valuetext="50% split"
          />
          <span class="readout" id="readout">50%</span>
        </div>
      </div>
      <script>
        const slider = document.getElementById('compare-slider');
        const stage = document.getElementById('stage');
        const readout = document.getElementById('readout');
        function syncSplit() {{
          const val = slider.value;
          slider.style.setProperty('--attr-min', slider.min || 0);
          slider.style.setProperty('--attr-max', slider.max || 100);
          slider.style.setProperty('--control-value', val);
          slider.setAttribute('aria-valuetext', val + '% split');
          stage.style.setProperty('--split', val + '%');
          readout.textContent = val + '%';
        }}
        syncSplit();
        slider.addEventListener('input', syncSplit);
      </script>
    </body>
    </html>
    """
    components.html(html, height=iframe_height, scrolling=False)


# ===============================
# Main Studio Application
# ===============================
def main():
    inject_studio_css()

    loaded_styles, device = load_style_models()
    predict_fn, preprocess_fn = load_segmentation_model()
    device_label = f"CUDA ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "CPU Engine"

    # -------------------------------
    # Sidebar: Studio Inspector
    # -------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-logo">✦</div>
                <div>
                    <div class="sidebar-brand-title">Persona-SegNST</div>
                    <div class="sidebar-brand-sub">v2.0 · Dual-Branch Studio</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-label">Compositing Mode</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Select Compositing Mode",
            options=list(MODE_CATALOG.keys()),
            format_func=lambda k: f"{MODE_CATALOG[k]['label']} ({k})",
            index=0,
            label_visibility="collapsed",
        )
        st.caption(MODE_CATALOG[mode]["desc"])

        st.markdown('<div class="sidebar-section-label">Neural Style Preset</div>', unsafe_allow_html=True)
        style_choice = st.selectbox(
            "Choose Style Preset",
            options=list(STYLE_CATALOG.keys()),
            index=0,
            disabled=(mode == "Segmentation"),
            label_visibility="collapsed",
        )
        style_meta = STYLE_CATALOG[style_choice]
        st.markdown(
            f"""
            <div class="style-preview-card">
                <div class="style-swatch-bar" style="background: {style_meta['gradient']};"></div>
                <div class="style-meta-row">
                    <span class="style-meta-name">{style_choice}</span>
                    <span class="style-meta-tag">{style_meta['tag']}</span>
                </div>
                <div class="style-meta-desc">{style_meta['desc']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section-label">Fine-Tuning & Matte</div>', unsafe_allow_html=True)
        style_strength = st.slider(
            "Style Intensity",
            min_value=0.0,
            max_value=1.0,
            value=1.0,
            step=0.05,
            disabled=(mode == "Segmentation"),
            help="Blend factor between stylized output and original image.",
        )
        threshold = st.slider(
            "Mask Threshold (τ)",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            disabled=(mode == "Style"),
            help="Confidence cutoff for separating foreground person from background.",
        )
        feather_px = st.slider(
            "Edge Feathering (px)",
            min_value=0,
            max_value=20,
            value=3,
            step=1,
            disabled=(mode == "Style"),
            help="Gaussian feather radius to soften silhouette edges.",
        )
        soft_matte = st.toggle(
            "Soft Probability Matte",
            value=True,
            disabled=(mode == "Style"),
            help="Uses continuous alpha transitions around the threshold instead of a hard binary cut.",
        )

        bg_hex = "#323232"
        if mode == "Segmentation":
            bg_hex = st.color_picker("Studio Backdrop Color", value="#323232")

        with st.expander("Engine & Resolution Settings", expanded=False):
            target_size = st.select_slider(
                "Canvas Max Resolution (px)",
                options=[384, 512, 640, 768],
                value=512,
            )
            preserve_aspect = st.toggle(
                "Preserve Aspect Ratio",
                value=True,
                help="Keeps natural photo proportions (dimensions snapped to multiples of 4).",
            )
            norm_mode = st.radio(
                "Segmentation Normalization",
                options=["Training [0, 1]", "MobileNetV2 [-1, 1]"],
                index=0,
                help="'Training [0, 1]' matches unet-3-model.ipynb normalization; 'MobileNetV2 [-1, 1]' matches sm.get_preprocessing.",
            )

    # -------------------------------
    # Main Studio Header
    # -------------------------------
    st.markdown(
        f"""
        <div class="studio-hero">
            <div>
                <h1>Persona-SegNST Studio</h1>
                <p>Semantic Person Segmentation &amp; Fast Neural Style Transfer with Spatial Alpha Compositing</p>
            </div>
            <div class="telemetry-row">
                <span class="telemetry-pill"><span class="telemetry-dot"></span>{device_label}</span>
                <span class="telemetry-pill"><span class="telemetry-dot violet"></span>MobileNetV2-UNet · 8.05M</span>
                <span class="telemetry-pill"><span class="telemetry-dot"></span>TransformerNet · 4 Styles</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop a portrait or full-body photo to enter the studio canvas",
        type=["jpg", "jpeg", "png"],
        help="Supports JPG, JPEG, and PNG formats.",
    )

    if uploaded_file is None:
        st.markdown(
            """
            <div class="mode-grid">
                <div class="mode-card">
                    <div class="mode-card-kicker">MODE 01 · StyleOnMask</div>
                    <div class="mode-card-title">Stylize Subject</div>
                    <div class="mode-card-desc">Transform only the segmented human silhouette into Mosaic, Candy, Rain, or Udnie art while preserving the real background.</div>
                </div>
                <div class="mode-card">
                    <div class="mode-card-kicker">MODE 02 · StyleOnBG</div>
                    <div class="mode-card-title">Stylize Background</div>
                    <div class="mode-card-desc">Keep facial features and clothing photorealistic while turning the entire surrounding environment into a painting.</div>
                </div>
                <div class="mode-card">
                    <div class="mode-card-kicker">MODE 03 · Style</div>
                    <div class="mode-card-title">Full-Frame NST</div>
                    <div class="mode-card-desc">Run single-pass feed-forward style transfer across the entire composition with controllable blend intensity.</div>
                </div>
                <div class="mode-card">
                    <div class="mode-card-kicker">MODE 04 · Segmentation</div>
                    <div class="mode-card-title">Subject Isolation</div>
                    <div class="mode-card-desc">Extract a clean person cutout onto any studio color backdrop or download a transparent RGBA PNG.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # -------------------------------
    # Decode & Execute Dual-Branch Pipeline
    # -------------------------------
    file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
    raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if raw_bgr is None:
        st.error("Unable to decode the uploaded image. Please upload a valid JPG or PNG file.")
        return

    frame_bgr = prepare_frame(raw_bgr, preserve_aspect=preserve_aspect, target_size=target_size)
    h, w = frame_bgr.shape[:2]

    with st.spinner("Running dual-branch neural pipeline..."):
        prob_map, seg_ms = run_segmentation(frame_bgr, predict_fn, preprocess_fn, norm_mode)
        alpha_3ch = build_alpha_matte(prob_map, threshold=threshold, feather_px=feather_px, soft_matte=soft_matte)

        if mode == "Segmentation":
            styled_bgr = None
            nst_ms = 0.0
        else:
            styled_bgr, nst_ms = stylize_frame(frame_bgr, loaded_styles[style_choice], device)

        result_bgr = composite_output(
            frame_bgr=frame_bgr,
            styled_bgr=styled_bgr,
            alpha_3ch=alpha_3ch,
            mode=mode,
            style_strength=style_strength,
            bg_hex=bg_hex,
        )

    orig_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    subject_pct = float((prob_map > threshold).mean() * 100.0)
    total_ms = seg_ms + nst_ms

    # -------------------------------
    # Live Telemetry Strip
    # -------------------------------
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Canvas Output</div>
                <div class="metric-value">{w} × {h}</div>
                <div class="metric-sub">{MODE_CATALOG[mode]['label']}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Subject Coverage</div>
                <div class="metric-value">{subject_pct:.1f}%</div>
                <div class="metric-sub">Threshold τ = {threshold:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Segmentation Latency</div>
                <div class="metric-value">{seg_ms:.0f} ms</div>
                <div class="metric-sub">{norm_mode}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Style / Total Latency</div>
                <div class="metric-value">{nst_ms:.0f} / {total_ms:.0f} ms</div>
                <div class="metric-sub">{'Bypassed' if mode == 'Segmentation' else style_choice + f' ({int(style_strength * 100)}%)'}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------
    # Interactive Studio Workspace Tabs
    # -------------------------------
    tab_split, tab_side, tab_diag, tab_gallery = st.tabs(
        ["◧ Split Compare", "▥ Side-by-Side", "◎ Mask & Diagnostics", "❖ 4-Style Contact Sheet"]
    )

    with tab_split:
        active_style_tag = "Backdrop" if mode == "Segmentation" else style_choice
        render_split_compare(orig_rgb, result_rgb, MODE_CATALOG[mode]["label"], active_style_tag)

    with tab_side:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown(
                '<div class="canvas-card-header"><span>ORIGINAL FRAME</span><span class="canvas-badge">INPUT</span></div>',
                unsafe_allow_html=True,
            )
            st.image(orig_rgb, use_container_width=True)
        with c2:
            st.markdown(
                f'<div class="canvas-card-header"><span>COMPOSITING RESULT</span><span class="canvas-badge">{mode} · {style_choice}</span></div>',
                unsafe_allow_html=True,
            )
            st.image(result_rgb, use_container_width=True)

    with tab_diag:
        heatmap_bgr = cv2.applyColorMap((prob_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
        matte_rgb = (alpha_3ch * 255).astype(np.uint8)
        tint = np.zeros_like(orig_rgb)
        tint[:, :, 0] = 124
        tint[:, :, 1] = 92
        tint[:, :, 2] = 255
        overlay_rgb = np.clip(orig_rgb * 0.65 + (tint * alpha_3ch) * 0.45, 0, 255).astype(np.uint8)

        d1, d2, d3 = st.columns(3, gap="medium")
        with d1:
            st.markdown(
                '<div class="canvas-card-header"><span>PROBABILITY MAP</span><span class="canvas-badge">RAW U-NET</span></div>',
                unsafe_allow_html=True,
            )
            st.image(heatmap_rgb, use_container_width=True)
        with d2:
            st.markdown(
                '<div class="canvas-card-header"><span>ALPHA MATTE</span><span class="canvas-badge">REFINED</span></div>',
                unsafe_allow_html=True,
            )
            st.image(matte_rgb, use_container_width=True)
        with d3:
            st.markdown(
                '<div class="canvas-card-header"><span>SILHOUETTE OVERLAY</span><span class="canvas-badge">COMPOSITE</span></div>',
                unsafe_allow_html=True,
            )
            st.image(overlay_rgb, use_container_width=True)

    with tab_gallery:
        if mode == "Segmentation":
            st.info("Switch Compositing Mode to StyleOnMask, StyleOnBG, or Style to generate a 4-style contact sheet.")
        else:
            st.caption(f"Render all four pretrained neural styles simultaneously under **{MODE_CATALOG[mode]['label']}**.")
            if st.button("Generate 4-Style Contact Sheet", use_container_width=False):
                cols = st.columns(2, gap="medium")
                for idx, (s_name, s_model) in enumerate(loaded_styles.items()):
                    s_bgr, s_ms = stylize_frame(frame_bgr, s_model, device)
                    comp_bgr = composite_output(frame_bgr, s_bgr, alpha_3ch, mode, style_strength, bg_hex)
                    comp_rgb = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB)
                    with cols[idx % 2]:
                        st.markdown(
                            f'<div class="canvas-card-header"><span>{s_name.upper()} · {STYLE_CATALOG[s_name]["tag"]}</span>'
                            f'<span class="canvas-badge">{s_ms:.0f} ms</span></div>',
                            unsafe_allow_html=True,
                        )
                        st.image(comp_rgb, use_container_width=True)

    # -------------------------------
    # Export Bar
    # -------------------------------
    st.markdown("---")
    ex1, ex2, ex3 = st.columns(3, gap="medium")
    base_name = os.path.splitext(uploaded_file.name)[0]
    with ex1:
        st.download_button(
            label="Download Result (PNG)",
            data=encode_image_bytes(result_rgb, "PNG"),
            file_name=f"{base_name}_{mode}_{style_choice}.png",
            mime="image/png",
            use_container_width=True,
        )
    with ex2:
        st.download_button(
            label="Download Result (JPEG)",
            data=encode_image_bytes(result_rgb, "JPEG"),
            file_name=f"{base_name}_{mode}_{style_choice}.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )
    with ex3:
        st.download_button(
            label="Download Transparent Cutout (RGBA PNG)",
            data=encode_image_bytes(orig_rgb, "RGBA_PNG", alpha_1ch=alpha_3ch[:, :, 0]),
            file_name=f"{base_name}_cutout_rgba.png",
            mime="image/png",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
