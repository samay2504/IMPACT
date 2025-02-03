import rasterio
from rasterio.windows import Window

def create_tiles(image_path, tile_size):
    with rasterio.open(image_path) as src:
        meta = src.meta.copy()
        for i in range(0, src.width, tile_size):
            for j in range(0, src.height, tile_size):
                window = Window(i, j, tile_size, tile_size)
                transform = src.window_transform(window)
                meta.update({"transform": transform, "width": window.width, "height": window.height})
                tile = src.read(window=window)
                yield tile, meta, i, j


# Path to the TIFF file
tiff_file = 'converted/ndvi_rgb.tif'
tile_size = 512  # Example tile size, adjust as needed

# Create tiles
tiles = list(create_tiles(tiff_file, tile_size))

# Example of how to process the tiles (e.g., saving them)
for tile, meta, i, j in tiles:
    tile_filename = f"tile_{i}_{j}.tif"
    with rasterio.open(tile_filename, 'w', **meta) as dst:
        dst.write(tile)
    print(f"Saved tile {tile_filename}")
