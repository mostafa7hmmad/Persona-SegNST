import cv2
import numpy as np
import torch
from config import STYLE_MODELS, preprocess_input
from models.transformer import TransformerNet
from models.segmentation import predict
from utils.stylize import stylize
from utils.keyboard import draw_keyboard, mouse_click, current_mode, current_button
import tensorflow as tf

def run_app():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load style models
    loaded_styles = {}
    for name, path in STYLE_MODELS.items():
        model = TransformerNet().to(device)
        state_dict = torch.load(path, map_location=device)
        for k in list(state_dict.keys()):
            if k.endswith(("running_mean", "running_var")):
                del state_dict[k]
        model.load_state_dict(state_dict)
        model.eval()
        loaded_styles[name] = model

    current_style = loaded_styles["Candy"]

    # Start webcam
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("App", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("App", 625, 625)
    cv2.setMouseCallback("App", mouse_click, {"loaded_styles": loaded_styles, "style_models": STYLE_MODELS})

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (625, 625))

        # segmentation
        input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_frame = cv2.resize(input_frame, (224, 224))
        input_frame = preprocess_input(input_frame)
        input_frame = np.expand_dims(input_frame, axis=0).astype(np.float32)
        pred_mask = predict(tf.constant(input_frame))[0].numpy()
        pred_mask = cv2.resize(pred_mask, (625, 625))
        pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
        mask_3ch = cv2.cvtColor(pred_mask, cv2.COLOR_GRAY2BGR)
        alpha = pred_mask.astype(np.float32) / 255.0
        alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

        # apply mode
        if current_mode == "Style":
            styled = stylize(frame, current_style, device)
        elif current_mode == "Segmentation":
            background = np.full_like(frame, (50, 50, 50))
            styled = (frame * alpha + background * (1 - alpha)).astype(np.uint8)
        elif current_mode == "StyleOnMask":
            mask_part = stylize(frame, current_style, device)
            styled = (mask_part * alpha + frame * (1 - alpha)).astype(np.uint8)
        elif current_mode == "StyleOnBG":
            bg_part = stylize(frame, current_style, device)
            styled = (frame * alpha + bg_part * (1 - alpha)).astype(np.uint8)

        styled = draw_keyboard(styled)
        cv2.imshow("App", styled)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
