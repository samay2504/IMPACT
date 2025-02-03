import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import box


def apply_shapefile_mask(raster_path, shapefile_path):
    # Read raster and shapefile
    with rasterio.open(raster_path) as src:
        # Get raster bounds
        raster_bounds = src.bounds
        print(f"Raster bounds: {raster_bounds}")

        # Load shapefile and check CRS
        boundary_gdf = gpd.read_file(shapefile_path)
        print(f"Shapefile CRS: {boundary_gdf.crs}")

        # Reproject shapefile to raster CRS if necessary
        if boundary_gdf.crs != src.crs:
            print("Shapefile reprojected to match raster CRS.")
            boundary_gdf = boundary_gdf.to_crs(src.crs)

        # Get shapefile bounds
        shapefile_bounds = boundary_gdf.total_bounds
        print(f"Shapefile bounds: {shapefile_bounds}")

        # Clip shapefile to raster bounds
        raster_bbox = box(*raster_bounds)
        boundary_gdf_clipped = boundary_gdf.clip(raster_bbox)

        # Spatial join to keep only intersecting geometries
        raster_bbox_gdf = gpd.GeoDataFrame({'geometry': [raster_bbox]}, crs=src.crs)
        boundary_gdf_clipped = gpd.sjoin(boundary_gdf_clipped, raster_bbox_gdf, how='inner', predicate='intersects')

        if boundary_gdf_clipped.empty:
            raise ValueError("No intersection between raster and shapefile after clipping.")

        # Apply mask to the clipped shapefile geometries
        out_image, out_transform = mask(src, boundary_gdf_clipped.geometry, crop=True)

        # Get metadata
        out_meta = src.meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        return out_image, out_meta


# File paths
raster_path = 'D:/Projects/Impact hp/Dataset1/nir train2.tif'
shapefile_path = 'D:/Projects/Impact hp/boundary.shp'

# Call the function and catch the result
ndvi_image, ndvi_meta = apply_shapefile_mask(raster_path, shapefile_path)
