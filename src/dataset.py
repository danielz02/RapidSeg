import os
import rasterio
import numpy as np
from torch.utils import data
from albumentations import Compose


class RapidEye(data.Dataset):
    def __init__(
            self, img_dir: str, mask_dir: str or None, band: str,
            transform: Compose = None  # Albumentations transformation objects
    ):
        super().__init__()
        assert band is "RGB" or "Full"

        self.band = band
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(os.listdir(self.img_dir))

    def __getitem__(self, index: int):
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[index])
        img_raster = rasterio.open(img_path)
        img_arr = img_raster.read()
        _, h, w = img_arr.shape
        img_raster.close()

        img_arr = img_arr.astype(np.longlong)
        if self.band is "RGB":
            img_arr = img_arr[0:3, :].copy()
        img_arr = img_arr.transpose(1, 2, 0)
        if not self.mask_dir:
            return (self.transform(image=img_arr)["image"], None) if self.transform else (img_arr, None)

        mask_path = os.path.join(self.mask_dir, os.listdir(self.mask_dir)[index])
        mask_raster = rasterio.open(mask_path)
        mask_arr = mask_raster.read()
        mask_raster.close()

        mask_arr = mask_arr.astype(np.longlong)
        mask_arr[np.logical_and(mask_arr != 1, mask_arr != 5)] = 0
        mask_arr[mask_arr == 5] = 2
        mask_arr = mask_arr.reshape(h, w)

        transformed = self.transform(image=img_arr, masks=[mask_arr])
        return (transformed["image"], transformed["masks"][0]) if self.transform else (img_arr, mask_arr)
