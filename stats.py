import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Load the TIFF files
tif_files = [
    'D:/Projects/Impact hp/Dataset1/z0.tif',
    'D:/Projects/Impact hp/Dataset1/z1.tif',
    'D:/Projects/Impact hp/Dataset1/z2.tif',
    'D:/Projects/Impact hp/Dataset1/z3.tif'
]


# Function to calculate NDVI statistics and sample values
def calculate_ndvi_stats_and_sample(tif_path, sample_size=10000):
    with rasterio.open(tif_path) as src:
        ndvi = src.read(1)  # Read the NDVI band (assuming it's the first band)

        # Calculate statistics
        ndvi_mean = np.mean(ndvi)
        ndvi_median = np.median(ndvi)
        ndvi_min = np.min(ndvi)
        ndvi_max = np.max(ndvi)

        print(f"File: {tif_path}")
        print(f"NDVI Mean: {ndvi_mean}")
        print(f"NDVI Median: {ndvi_median}")
        print(f"NDVI Min: {ndvi_min}")
        print(f"NDVI Max: {ndvi_max}")
        print()

        # Flatten the NDVI values and take a random sample
        ndvi_flat = ndvi.flatten()
        sample_indices = np.random.choice(len(ndvi_flat), size=sample_size, replace=False)
        sampled_ndvi = ndvi_flat[sample_indices]

        return sampled_ndvi


# Calculate and display NDVI statistics and sample values for each file
sampled_ndvi_values = []
for tif_file in tif_files:
    sampled_ndvi = calculate_ndvi_stats_and_sample(tif_file)
    sampled_ndvi_values.extend(sampled_ndvi)

# Convert to numpy array for efficient computation
sampled_ndvi_values = np.array(sampled_ndvi_values)

# Plot histogram of sampled NDVI values
'''
plt.hist(sampled_ndvi_values, bins=50, alpha=0.7)
plt.title('NDVI Histogram for P. Juliflora')
plt.xlabel('NDVI Value')
plt.ylabel('Frequency')
plt.show()
'''
hist, bin_edges = np.histogram(sampled_ndvi_values, bins=50)

bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.bar(bin_centers, hist, width=(bin_edges[1] - bin_edges[0]), alpha=0.7)

# Add title and labels
plt.title('NDVI Bar Graph for P. Juliflora')
plt.xlabel('NDVI Value')
plt.ylabel('Frequency')

# Show the plot
plt.show()