import tensorflow as tf
from tensorflow import keras
from config import MODEL_PATH

seg_model = keras.models.load_model(MODEL_PATH, compile=False)

@tf.function
def predict(input_tensor):
    return seg_model(input_tensor, training=False)
