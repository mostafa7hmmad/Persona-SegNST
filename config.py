import os
import types
import patch_keras  # ✅ patch keras before segmentation_models

import segmentation_models as sm
import tensorflow as tf
from tensorflow import keras

# Patch for generic_utils bug
if not hasattr(keras.utils, "generic_utils"):
    keras.utils.generic_utils = types.SimpleNamespace()
    keras.utils.generic_utils.to_list = lambda x: x if isinstance(x, list) else [x]
    keras.utils.generic_utils.get_custom_objects = lambda: {}

os.environ["SM_FRAMEWORK"] = "tf.keras"

# Segmentation model config
MODEL_PATH = "best_Unet.h5"
BACKBONE = "mobilenetv2"
preprocess_input = sm.get_preprocessing(BACKBONE)

# Style model paths
STYLE_MODELS = {
    "Candy": "candy.pth",
    "Mosaic": "mosaic.pth",
    "Rain": "rain_princess.pth",
    "Udnie": "udnie.pth"
}

# Virtual keyboard
GROUPS = {
    "Styles": list(STYLE_MODELS.keys()),
    "Seg": ["Segmentation"],
    "Seg+Styles": ["StyleOnMask", "StyleOnBG"]
}
KEY_W, KEY_H = 70, 30
START_X, START_Y = 10, 20
