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
        """
        Instantiate a dataloader for the RapidEye dataset
        :param img_dir: The path to the directory of GeoTiff images to load
        :param mask_dir: The path to the directory of GeoTiff mask to load
        :param band: The number of bands to load (RGB or Full bands)
        :param transform: Object that implements torchvision.transforms API
        """
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
        _, h, w = img_arr.shape  # Rasterio loads raster in CHW format
        img_raster.close()

        # Convert to int64 so that transformations can be applied as required by albumentations
        img_arr = img_arr.astype(np.longlong)
        if self.band is "RGB":
            img_arr = img_arr[0:3, :].copy()
        img_arr = img_arr.transpose(1, 2, 0)  # Convert to HWC
        if not self.mask_dir:
            return (self.transform(image=img_arr)["image"], None) if self.transform else (img_arr, None)

        mask_path = os.path.join(self.mask_dir, os.listdir(self.mask_dir)[index])
        mask_raster = rasterio.open(mask_path)
        mask_arr = mask_raster.read()
        mask_raster.close()

        # Encode new labels: 0 -> Others | 1 -> Corn | 2 -> Soybean
        mask_arr = mask_arr.astype(np.longlong)
        mask_arr[np.logical_and(mask_arr != 1, mask_arr != 5)] = 0
        mask_arr[mask_arr == 5] = 2
        mask_arr = mask_arr.reshape(h, w)

        transformed = self.transform(image=img_arr, masks=[mask_arr])  # data augmentation for image and mask
        return (transformed["image"], transformed["masks"][0]) if self.transform else (img_arr, mask_arr)
