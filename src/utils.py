import os
import rasterio
import numpy as np
from torch.utils import data
from torchvision.transforms import transforms


class RapidEye(data.Dataset):
    def __init__(self, img_dir: str, mask_dir: str or None, transform: transforms = None):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(os.listdir(self.img_dir))

    def __getitem__(self, index: int):
        img_path = os.path.join(self.img_dir, os.listdir(self.img_dir)[index])
        img_raster = rasterio.open(img_path)
        img_arr = img_raster.read()
        img_raster.close()
        if not self.mask_dir:
            return self.transform(img_arr), None if self.transform else img_arr, None

        mask_path = os.path.join(self.mask_dir, os.listdir(self.mask_dir)[index])
        mask_raster = rasterio.open(mask_path)
        mask_arr = mask_raster.read()
        mask_raster.close()
        mask_arr[np.logical_and(mask_arr != 1, mask_arr != 5)] = 0

        return self.transform(img_arr, mask_arr) if self.transform else img_arr, mask_arr
