import types
from tensorflow import keras

if not hasattr(keras.utils, "generic_utils"):
    keras.utils.generic_utils = types.SimpleNamespace()
    keras.utils.generic_utils.to_list = lambda x: x if isinstance(x, list) else [x]
    keras.utils.generic_utils.get_custom_objects = lambda: {}
