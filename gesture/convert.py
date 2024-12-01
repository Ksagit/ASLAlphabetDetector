# convert_model.py
import tensorflowjs as tfjs
import tensorflow as tf
import json
import numpy as np

# Load your Keras model
model = tf.keras.models.load_model('gesture_classifier_model.keras')

# Convert and save the model
tfjs.converters.save_keras_model(model, 'static/models')

# Save label mapping
data = np.load('preprocessed_data.npz', allow_pickle=True)
label_mapping = data['label_mapping'].item()

with open('static/models/label_mapping.json', 'w') as f:
    json.dump(label_mapping, f)