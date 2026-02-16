import os
import tensorflow as tf
from tf_keras.models import model_from_json
import json

print("TensorFlow version:", tf.__version__)
try:
    import tf_keras
    print("tf_keras version:", tf_keras.__version__)
except ImportError:
    print("tf_keras not found")

json_path = 'Models/model.json'
if os.path.exists(json_path):
    with open(json_path, 'r') as f:
        json_content = f.read()
    try:
        model = model_from_json(json_content)
        print("Model loaded successfully from JSON using tf_keras")
        weights_path = 'Models/model_weights.h5'
        if os.path.exists(weights_path):
            model.load_weights(weights_path)
            print("Weights loaded successfully")
        else:
             print("Weights file not found")
    except Exception as e:
        print("Error loading model:", e)
        import traceback
        traceback.print_exc()
else:
    print("JSON file not found")
