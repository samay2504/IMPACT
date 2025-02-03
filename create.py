import tensorflow as tf
from tensorflow.keras import layers, models
import json
def create_model_from_tflite_architecture(input_shape, model_details):
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    for layer in model_details['layers']:
        # Based on the extracted layer info, add appropriate layers
        if 'Conv2D' in layer['name']:
            x = layers.Conv2D(filters=layer['shape'][-1], kernel_size=(3, 3), padding='same', activation='relu')(x)
        elif 'Dense' in layer['name']:
            x = layers.Dense(units=layer['shape'][-1], activation='relu')(x)
        # Add other layer types as needed

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# Load model details
with open('model_details.json', 'r') as f:
    model_details = json.load(f)

# Assume input shape is from the first input tensor of the TFLite model
input_shape = tuple(model_details['inputs'][0]['shape'][1:])
model = create_model_from_tflite_architecture(input_shape, model_details)

# Print model summary
model.summary()
