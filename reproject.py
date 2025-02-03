import rasterio
import geopandas as gpd

# Check CRS of raster
with rasterio.open('D:/Projects/Impact hp/Dataset1/nir train2.tif') as src:
    raster_crs = src.crs
    print("Raster CRS:", raster_crs)

# Check CRS of shapefile
boundary_gdf = gpd.read_file('D:/Projects/Impact hp/Boundary.shp')
shapefile_crs = boundary_gdf.crs
print("Shapefile CRS:", shapefile_crs)

# If they don't match, reproject the shapefile to match the raster
if shapefile_crs != raster_crs:
    boundary_gdf = boundary_gdf.to_crs(raster_crs)
    print("Shapefile reprojected to match raster CRS.")

# Check bounds
with rasterio.open('D:/Projects/Impact hp/Dataset1/nir train2.tif') as src:
    raster_bounds = src.bounds
    print("Raster bounds:", raster_bounds)

shapefile_bounds = boundary_gdf.total_bounds
print("Shapefile bounds:", shapefile_bounds)
