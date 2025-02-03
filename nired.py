import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np

# Load boundary shapefile
boundary_gdf = gpd.read_file('D:/Projects/Impact hp/Boundary.shp')

# Reproject the shapefile to match the raster CRS (EPSG:32643)
def reproject_shapefile(shapefile, crs):
    return shapefile.to_crs(crs)

# Load .tif image and apply mask using shapefile
def apply_shapefile_mask(image_path, shapefile):
    with rasterio.open(image_path) as src:
        # Reproject shapefile to match the raster CRS if necessary
        shapefile = reproject_shapefile(shapefile, src.crs)
        out_image, out_transform = mask(src, shapefile.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
    return out_image, out_meta

# Apply the shapefile mask separately to NIR and Red .tif files
nir_image, nir_meta = apply_shapefile_mask("C:/Users/Samay Mehar/Downloads/NIR.TIF", boundary_gdf)
red_image, red_meta = apply_shapefile_mask("C:/Users/Samay Mehar/Downloads/red.TIF", boundary_gdf)

# Ensure the dimensions match after masking
if nir_image.shape != red_image.shape:
    raise ValueError("The NIR and Red images do not have the same dimensions after masking!")

# NDVI calculation (Assuming nir_image and red_image are single-band images)
def calculate_ndvi(nir_image, red_image, epsilon=1e-10):
    nir = nir_image[0]  # First (and only) band of NIR image
    red = red_image[0]  # First (and only) band of Red image
    ndvi = (nir - red) / (nir + red + epsilon)  # Add epsilon to avoid division by zero
    return ndvi


# Calculate NDVI for the masked images
ndvi = calculate_ndvi(nir_image, red_image)

# Convert NDVI to binary mask based on a threshold
def create_binary_mask(ndvi, threshold=0.3):
    mask = np.where(ndvi > threshold, 1, 0)  # Threshold to detect P. juliflora
    return mask

# Generate binary mask for the NDVI
binary_mask = create_binary_mask(ndvi)

# Create red-white output image based on binary mask
def generate_output_image(binary_mask):
    h, w = binary_mask.shape
    output_image = np.zeros((h, w, 3), dtype=np.uint8)
    output_image[binary_mask == 1] = [255, 0, 0]  # Red for detected P. juliflora
    output_image[binary_mask == 0] = [255, 255, 255]  # White for non-detected areas
    return output_image

# Generate the final output image
output_image = generate_output_image(binary_mask)

# Save the output image as .tif
with rasterio.open('D:/Projects/Impact hp/Result/output_image.tif', 'w', driver='GTiff',
                   height=output_image.shape[0], width=output_image.shape[1],
                   count=3, dtype=output_image.dtype, crs=nir_meta['crs'],
                   transform=nir_meta['transform']) as dst:
    dst.write(output_image.transpose(2, 0, 1))  # Transpose to (C, H, W) format


print("Output image saved as 'output_image.tif'")
