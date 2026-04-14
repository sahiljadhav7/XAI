from keras.models import load_model
from keras.layers import Dense
import os

class CustomDense(Dense):
    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)
        return super().from_config(config)

model_path = os.path.join("d:\\XAI\\XAI\\static\\models\\lung_disease", "model.h5")
print("Loading model...")
try:
    model = load_model(model_path, custom_objects={'Dense': CustomDense})
    print("Model loaded successfully!")
except Exception as e:
    print("Error:", e)
