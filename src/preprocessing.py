import os
import rasterio
from params import *
from tqdm import tqdm
from rasterio import windows
from itertools import product


def get_tile(f: rasterio.DatasetReader, row: int, col: int, w: int, h: int) -> (windows.Window, rasterio.Affine):
    """
    Chop a tile with given width and height starting from a certain pixel coordinate pair
    :param f: a rasterio DatasetReader object
    :param row: row offset (in pixel) of the top left corner
    :param col: column offset (in pixel) of the top left corner
    :param w: width (in the number of pixels) of the tile
    :param h: height (in the number of pixels) of the tile
    :return: a Window object of the tile window
    """
    tile_window = windows.Window(col_off=col, row_off=row, width=w, height=h)
    t = windows.transform(tile_window, f.transform)
    return tile_window, t


def create_tiles(src_folder: str, raster_name: str, tile_dest: str, tile_width: int, tile_height: int):
    """
    Crop a raster file into tiles
    :param src_folder: path to the folder of dataset
    :param raster_name: name of the raster file to be processed
    :param tile_dest: name of the folder for cropped tiles
    :param tile_width: width (in the number of pixels) of the tile
    :param tile_height: height (in the number of pixels) of the tile
    :return: nothing
    """
    if not os.path.exists(f"../data/{tile_dest}"):
        os.mkdir(f"../data/{tile_dest}")
    with rasterio.open(os.path.join(src_folder, raster_name), "r") as f_in:
        raster_width, raster_height = f_in.meta["width"], f_in.meta["height"]
        for row, col in tqdm(product(range(0, raster_height, tile_height),
                                     range(0, raster_width, tile_width))):
            tile, transform = get_tile(f_in, row, col, tile_width, tile_height)
            meta = f_in.meta.copy()
            meta["transform"] = transform
            meta["width"], meta["height"] = tile.width, tile.height
            with rasterio.open(os.path.join(src_folder, tile_dest, f"{row}-{col}.tif"), "w", **meta) as f_out:
                f_out.write(f_in.read(window=tile))


if __name__ == "__main__":
    create_tiles(data_path, test_file, "test", 256, 256)
    create_tiles(data_path, train_file, "train", 256, 256)
    create_tiles(data_path, mask_file, "train_mask", 256, 256)
