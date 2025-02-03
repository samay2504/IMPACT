import tensorflow as tf
import rasterio
import numpy as np
import json
import cv2
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, UpSampling2D

# Load the model configuration
with open('D:/Projects/Impact hp/config.json', 'r') as f:
    model_config = json.load(f)


class SpatialPyramidPooling(tf.keras.layers.Layer):
    def __init__(self, dilation_rates, num_channels, **kwargs):
        super(SpatialPyramidPooling, self).__init__(**kwargs)
        self.dilation_rates = dilation_rates
        self.num_channels = num_channels
        self.conv_layers = [
            Conv2D(num_channels, (3, 3), dilation_rate=rate, padding='same', activation='relu')
            for rate in dilation_rates
        ]

    def call(self, inputs):
        pooled = [conv(inputs) for conv in self.conv_layers]
        return tf.concat(pooled, axis=-1)


# Reconstruct the segmentation head
def build_segmentation_head(head_config):
    layers = []
    for layer_config in head_config['config']['layers']:
        if layer_config['class_name'] == 'Conv2D':
            layers.append(Conv2D(**layer_config['config']))
        elif layer_config['class_name'] == 'BatchNormalization':
            layers.append(BatchNormalization(**layer_config['config']))
        elif layer_config['class_name'] == 'ReLU':
            layers.append(ReLU(**layer_config['config']))
        elif layer_config['class_name'] == 'UpSampling2D':
            layers.append(UpSampling2D(**layer_config['config']))
        elif layer_config['class_name'] == 'InputLayer':
            continue  # InputLayer is already handled
        else:
            raise ValueError(f"Unsupported layer type: {layer_config['class_name']}")
    return tf.keras.Sequential(layers)


# Create the base model (DeepLabV3Plus)
def build_model(model_config):
    num_classes = model_config['config']['num_classes']

    # Manually build ResNetV2 backbone
    backbone = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(512, 512, 3), weights=None)

    # Build spatial pyramid pooling
    spp_config = model_config['config']['spatial_pyramid_pooling']
    spp_layer = SpatialPyramidPooling(**spp_config['config'])

    # Build segmentation head
    head_config = model_config['config']['segmentation_head']
    segmentation_head = build_segmentation_head(head_config)

    inputs = tf.keras.Input(shape=(512, 512, 3))
    x = backbone(inputs)
    x = spp_layer(x)
    outputs = segmentation_head(x)

    model = tf.keras.Model(inputs, outputs)
    return model


# Reconstruct the model
model = build_model(model_config)

# Load the weights
model.load_weights('D:/Projects/Impact hp/model.weights.h5')


# Prepare dataset for fine-tuning
def extract_ndvi(tif_path):
    with rasterio.open(tif_path) as src:
        ndvi = src.read(1)  # Assuming NDVI values are in the first band
    ndvi_resized = cv2.resize(ndvi, (512, 512), interpolation=cv2.INTER_LINEAR)  # Resize to fixed size
    return ndvi_resized


def prepare_dataset(tif_files):
    X = []
    for tif in tif_files:
        ndvi = extract_ndvi(tif)
        ndvi_3channel = np.stack((ndvi, ndvi, ndvi), axis=-1)  # Replicate to three channels
        X.append(ndvi_3channel)
    return np.array(X)


# Assuming you have mask data in the following files
mask_files = [
    'D:/Projects/Impact hp/Dataset1/mask_z0.tif',
    'D:/Projects/Impact hp/Dataset1/mask_z1.tif',
    'D:/Projects/Impact hp/Dataset1/mask_z2.tif',
    'D:/Projects/Impact hp/Dataset1/mask_z3.tif'
]


def prepare_mask(mask_files):
    y = []
    for mask in mask_files:
        with rasterio.open(mask) as src:
            mask_data = src.read(1)  # Assuming mask data is in the first band
        mask_resized = cv2.resize(mask_data, (512, 512), interpolation=cv2.INTER_NEAREST)  # Resize to fixed size
        y.append(mask_resized)
    y = np.expand_dims(np.array(y), axis=-1)  # Add channel dimension
    return y


# Example Usage
tif_files = [
    'D:/Projects/Impact hp/Dataset1/z0.tif',
    'D:/Projects/Impact hp/Dataset1/z1.tif',
    'D:/Projects/Impact hp/Dataset1/z2.tif',
    'D:/Projects/Impact hp/Dataset1/z3.tif'
]
X = prepare_dataset(tif_files)
y = prepare_mask(mask_files)

# Split into training and validation sets (assuming you have labeled mask data)
X_train, X_val = X[:3], X[3:]
y_train, y_val = y[:3], y[3:]


# Fine-tune the model
def fine_tune_model(model, X_train, y_train, X_val, y_val, num_classes=1):
    # Add a final layer with num_classes outputs
    outputs = Conv2D(num_classes, (1, 1), activation='sigmoid')(model.output)
    model = tf.keras.Model(inputs=model.input, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=16)

    # Save the fine-tuned model
    model.save('D:/Projects/Impact hp/new model/fine_tuned_deeplabv3_plus_resnet50_pascalvoc.h5')
    return model


# Fine-tune the model
model = fine_tune_model(model, X_train, y_train, X_val, y_val)


# Predict and create masked TIFF files
def predict_and_create_tif(model, tif_path, output_path):
    ndvi = extract_ndvi(tif_path)
    ndvi_3channel = np.stack((ndvi, ndvi, ndvi), axis=-1)  # Replicate to three channels
    ndvi_3channel = np.expand_dims(ndvi_3channel, axis=0)  # Add batch dimension
    prediction = model.predict(ndvi_3channel)
    prediction = prediction.squeeze()  # Remove batch dimension

    # Save prediction as a new TIFF file
    with rasterio.open(tif_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, count=1)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(prediction, 1)


# Predict and create output TIFF files
predict_and_create_tif(model, 'D:/Projects/Impact hp/Result/predict_ndvi.tif',
                       'D:/Projects/Impact hp/Result/output_mask.tif')
