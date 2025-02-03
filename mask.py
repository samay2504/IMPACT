import rasterio
import geopandas as gpd
from rasterio.features import rasterize

# Load the KML file
kml_path = r'D:/Projects/Impact hp/Prosopis juliflora infested sites for ground truthing.kml'

# Read the KML file using geopandas
gdf = gpd.read_file(kml_path, driver='KML')


# Create a function to create masks
def create_mask(tif_path, gdf, output_path):
    with rasterio.open(tif_path) as src:
        profile = src.profile
        transform = src.transform
        crs = src.crs

        # Reproject the GeoDataFrame to match the CRS of the raster
        gdf = gdf.to_crs(crs)

        # Rasterize the geometries
        mask = rasterize(
            [(geometry, 1) for geometry in gdf.geometry],
            out_shape=(src.height, src.width),
            transform=transform,
            fill=0,
            dtype=rasterio.uint8
        )

        # Update the profile and write the mask to a new TIF file
        profile.update(dtype=rasterio.uint8, count=1)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(mask, 1)


# Example usage
tif_files = [
    r'D:/Projects/Impact hp/Dataset1/z0.tif',
    r'D:/Projects/Impact hp/Dataset1/z1.tif',
    r'D:/Projects/Impact hp/Dataset1/z2.tif',
    r'D:/Projects/Impact hp/Dataset1/z3.tif'
]

for i, tif_file in enumerate(tif_files):
    create_mask(tif_file, gdf, f'D:/Projects/Impact hp/Dataset1/mask_z{i}.tif')
