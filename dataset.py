import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt


# Function to load single-band NDVI .tif images
def load_single_band_tif(path):
    try:
        with rasterio.open(path) as src:
            band = src.read(1)  # Read the first (and only) band
        return band
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


# Load dataset function adapted for single-band NDVI images
def load_ndvi_dataset(data_path):
    ndvi_images = []
    masks = []
    for file_name in os.listdir(data_path):
        if file_name.endswith('.tif') and not file_name.startswith('mask_z0'):
            file_path = os.path.join(data_path, file_name)
            mask_path = os.path.join(data_path, f"mask_{file_name}")

            if os.path.exists(mask_path):
                ndvi = load_single_band_tif(file_path)
                mask = load_single_band_tif(mask_path)

                if ndvi is not None and mask is not None:
                    ndvi = np.expand_dims(ndvi, axis=-1)  # Add channel dimension
                    mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
                    ndvi_images.append(ndvi)
                    masks.append(mask)
                else:
                    print(f"Skipping {file_name} due to loading issues.")
            else:
                print(f"Mask file not found for {file_name}")

    ndvi_images = np.array(ndvi_images)
    masks = np.array(masks)
    return ndvi_images, masks


# Paths to the dataset
train_data_path = 'D:/Projects/Impact hp/Dataset1/'

# Load datasets
x_train, y_train = load_ndvi_dataset(train_data_path)

print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Visualize one of the loaded NDVI images and its corresponding mask
if x_train.shape[0] > 0:
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(x_train[0, :, :, 0], cmap='RdYlGn')
    plt.title("NDVI Image")

    plt.subplot(1, 2, 2)
    plt.imshow(y_train[0, :, :, 0], cmap='gray')
    plt.title("Mask Image")

    plt.show()
else:
    print("No training data found.")
