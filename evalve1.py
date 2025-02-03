import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import rasterio
from rasterio.plot import reshape_as_image

# Custom Dataset class for loading data
class NDVIDataset(Dataset):
    def __init__(self, data_path):
        self.ndvi_images, self.masks = self.load_dataset(data_path)

    def load_tif_image(self, path):
        with rasterio.open(path) as src:
            bands = src.read()
            image = reshape_as_image(bands)
        return image

    def calculate_ndvi(self, image):
        red = image[:, :, 0].astype(float)
        nir = image[:, :, 3].astype(float)
        ndvi = (nir - red) / (nir + red)
        ndvi = np.nan_to_num(ndvi)  # Replace nan values with 0
        return ndvi

    def load_dataset(self, data_path):
        ndvi_images = []
        masks = []
        for file_name in os.listdir(data_path):
            if file_name.endswith('.tif') and 'mask' not in file_name:
                file_path = os.path.join(data_path, file_name)
                mask_path = file_path.replace('.tif', '_mask.tif')
                if os.path.exists(mask_path):
                    image = self.load_tif_image(file_path)
                    ndvi = self.calculate_ndvi(image)
                    ndvi = np.expand_dims(ndvi, axis=-1)  # Add channel dimension
                    ndvi_images.append(ndvi)
                    mask = self.load_tif_image(mask_path)
                    mask = mask[:, :, 0]  # Assuming the mask is a single channel image
                    mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
                    masks.append(mask)

        ndvi_images = np.array(ndvi_images)
        masks = np.array(masks)
        return ndvi_images, masks

    def __len__(self):
        return len(self.ndvi_images)

    def __getitem__(self, idx):
        ndvi = torch.tensor(self.ndvi_images[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)
        return ndvi.permute(2, 0, 1), mask.permute(2, 0, 1)  # Change to (C, H, W) format

# Define the model architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(8, 48, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(48, 12, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(12, 192, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(192, 48, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(48, 288, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(288, 48, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(48, 288, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(288, 80, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(80, 480, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(480, 80, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(80, 480, kernel_size=3, padding=1)
        self.conv14 = nn.Conv2d(480, 80, kernel_size=3, padding=1)
        self.conv15 = nn.Conv2d(80, 480, kernel_size=3, padding=1)
        self.conv16 = nn.Conv2d(480, 160, kernel_size=3, padding=1)
        self.conv17 = nn.Conv2d(160, 72, kernel_size=3, padding=1)
        self.conv18 = nn.Conv2d(72, 12, kernel_size=3, padding=1)
        self.conv19 = nn.Conv2d(12, 72, kernel_size=3, padding=1)
        self.conv20 = nn.Conv2d(72, 16, kernel_size=3, padding=1)
        self.conv21 = nn.Conv2d(16, 96, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(96, 16, kernel_size=3, padding=1)
        self.conv23 = nn.Conv2d(16, 96, kernel_size=3, padding=1)
        self.conv24 = nn.Conv2d(96, 16, kernel_size=3, padding=1)
        self.conv25 = nn.Conv2d(16, 96, kernel_size=3, padding=1)
        self.conv26 = nn.Conv2d(96, 32, kernel_size=3, padding=1)
        self.conv27 = nn.Conv2d(32, 192, kernel_size=3, padding=1)
        self.conv28 = nn.Conv2d(192, 32, kernel_size=3, padding=1)
        self.conv29 = nn.Conv2d(32, 192, kernel_size=3, padding=1)
        self.conv30 = nn.Conv2d(192, 32, kernel_size=3, padding=1)
        self.conv31 = nn.Conv2d(32, 192, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(192, 32, kernel_size=3, padding=1)
        self.conv33 = nn.Conv2d(32, 256, kernel_size=3, padding=1)
        self.conv34 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv35 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv36 = nn.Conv2d(256, 21, kernel_size=3, padding=1)
        self.conv37 = nn.Conv2d(21, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = torch.relu(self.conv7(x))
        x = torch.relu(self.conv8(x))
        x = torch.relu(self.conv9(x))
        x = torch.relu(self.conv10(x))
        x = torch.relu(self.conv11(x))
        x = torch.relu(self.conv12(x))
        x = torch.relu(self.conv13(x))
        x = torch.relu(self.conv14(x))
        x = torch.relu(self.conv15(x))
        x = torch.relu(self.conv16(x))
        x = torch.relu(self.conv17(x))
        x = torch.relu(self.conv18(x))
        x = torch.relu(self.conv19(x))
        x = torch.relu(self.conv20(x))
        x = torch.relu(self.conv21(x))
        x = torch.relu(self.conv22(x))
        x = torch.relu(self.conv23(x))
        x = torch.relu(self.conv24(x))
        x = torch.relu(self.conv25(x))
        x = torch.relu(self.conv26(x))
        x = torch.relu(self.conv27(x))
        x = torch.relu(self.conv28(x))
        x = torch.relu(self.conv29(x))
        x = torch.relu(self.conv30(x))
        x = torch.relu(self.conv31(x))
        x = torch.relu(self.conv32(x))
        x = torch.relu(self.conv33(x))
        x = torch.relu(self.conv34(x))
        x = torch.relu(self.conv35(x))
        x = torch.relu(self.conv36(x))
        x = self.sigmoid(self.conv37(x))
        return x

# Create the model
model = ConvNet()

# Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Paths to the dataset
train_data_path = 'D:/Projects/Impact hp/Dataset1/'

# Load datasets
train_dataset = NDVIDataset(train_data_path)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

# Training loop
for epoch in range(5):
    for ndvi, mask in train_loader:
        # Forward pass
        outputs = model(ndvi)
        loss = criterion(outputs, mask)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/5], Loss: {loss.item()}')

# Saving the PyTorch model
torch.save(model.state_dict(), 'D:/Projects/Impact hp/Result/model.pth')

print("Model has been successfully saved.")
