import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import rasterio
from rasterio.mask import mask
import geopandas as gpd
import torchvision.transforms as transforms


# Custom dataset class for satellite images (NDVI patches)
class SatelliteDataset(Dataset):
    def __init__(self, image, labels, transform=None):
        self.image = image
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# Load boundary shapefile
boundary_gdf = gpd.read_file('D:/Projects/Impact hp/Boundary.shp')

# Load .tif image and apply mask using shapefile
def apply_shapefile_mask(image_path, shapefile):
    with rasterio.open(image_path) as src:
        out_image, out_transform = mask(src, shapefile.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
    return out_image, out_meta

# Masking NDVI image
ndvi_image, ndvi_meta = apply_shapefile_mask('D:/Projects/Impact hp/Dataset1/nir train2.tif', boundary_gdf)

# NDVI calculation (Assuming bands 4 and 3 correspond to NIR and Red respectively)
#ndvi_image = (ndvi_image[3] - ndvi_image[2]) / (ndvi_image[3] + ndvi_image[2])

# Patch generation
def create_patches(ndvi, patch_size=32):
    patches = []
    labels = []
    h, w = ndvi.shape
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = ndvi[i:i + patch_size, j:j + patch_size]
            # Assuming label generation based on some threshold value for NDVI to detect P. juliflora
            label = 1 if np.mean(patch) > 0.3 else 0  # Threshold chosen as 0.3 for this example
            patches.append(patch)
            labels.append(label)
    return np.array(patches), np.array(labels)


# Generate patches from the masked NDVI image
patches, labels = create_patches(ndvi_image)

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Single channel normalization for NDVI
])

# Dataset and DataLoader
dataset = SatelliteDataset(patches, labels, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model (Using pretrained EfficientNet as a base, modifying for single-channel input)
model = models.efficientnet_b0(pretrained=True)
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # Adjust for single-channel input (NDVI)
model.classifier[1] = nn.Linear(in_features=model.classifier[1].in_features, out_features=2)  # Binary classification

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader)}")

# Inference on the entire NDVI image
model.eval()
with torch.no_grad():
    h, w = ndvi_image.shape
    patch_size = 32
    output_image = np.zeros((h, w, 3), dtype=np.uint8)  # RGB output image
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = ndvi_image[i:i + patch_size, j:j + patch_size]
            patch_tensor = transform(patch).unsqueeze(0).to(device)
            output = model(patch_tensor)
            _, predicted = torch.max(output, 1)
            if predicted.item() == 1:  # Detected *P. juliflora*
                output_image[i:i + patch_size, j:j + patch_size] = [255, 0, 0]  # Red for detected
            else:
                output_image[i:i + patch_size, j:j + patch_size] = [255, 255, 255]  # White for non-detected

# Save output .tif image
with rasterio.open('output.tif', 'w', driver='GTiff', height=output_image.shape[0], width=output_image.shape[1],
                   count=3, dtype=output_image.dtype) as dst:
    dst.write(output_image.transpose(2, 0, 1))  # Transpose to match (C, H, W) format

# Evaluate performance (assuming you have ground truth labels for comparison)
true_labels = labels  # Replace with actual ground truth labels for the patches
pred_labels = []
for images, _ in dataloader:
    images = images.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    pred_labels.extend(preds.cpu().numpy())

# Confusion matrix and accuracy
conf_matrix = confusion_matrix(true_labels, pred_labels)
accuracy = accuracy_score(true_labels, pred_labels)

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy: ", accuracy)
