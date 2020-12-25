import torch
import numpy as np
import matplotlib.pyplot as plt
from src.params import COLOR_MAP
from skimage.color import label2rgb


def normalize(arr: np.ndarray or torch.Tensor):
    """
    Normalize a single channel to [0, 1]
    :param arr: A slice of the channel to normalize
    :return: normalized channel
    """
    arr_min, arr_max = np.min(arr), np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def visualize_preprocess(
    img_batch: np.ndarray or torch.Tensor,
    mask_batch_target: np.ndarray or torch.Tensor,
    mask_batch_pred: np.ndarray or torch.Tensor = None
):
    """
    Take in a batch of images in NCHW, choose one image, and convert to HWC
    :param img_batch: the batch of images (in NCHW) to preprocess
    :param mask_batch_target: the batch of masks (in NHW)
    :param mask_batch_pred: the batch of predicted masks (in NHW)
    :return:
    """
    if isinstance(img_batch, torch.Tensor):
        img_batch = img_batch.cpu().numpy()
    if isinstance(mask_batch_target, torch.Tensor):
        mask_batch_target = mask_batch_target.cpu().numpy()
    if isinstance(mask_batch_pred, torch.Tensor):
        mask_batch_pred = mask_batch_pred.cpu().numpy()

    n, c, h, w = img_batch.shape
    rand_idx = np.random.randint(0, n)
    random_img = img_batch[rand_idx, :]
    random_target = mask_batch_target[rand_idx, :]

    # Keep RGB channels only
    if c > 3:
        random_img = random_img[:3, :]
    random_img[0, :] = normalize(random_img[0, :])
    random_img[1, :] = normalize(random_img[1, :])
    random_img[2, :] = normalize(random_img[2, :])
    random_img = random_img.transpose(1, 2, 0)  # matplotlib uses channel-last order

    if mask_batch_pred is not None:
        random_pred = mask_batch_pred[rand_idx, :]
        return random_img, random_target, random_pred
    else:
        return random_img, random_target


def visualize_sample(
    img_batch: np.ndarray or torch.Tensor,
    mask_batch_target: np.ndarray or torch.Tensor,
):
    """
    Display one image in the batch with mask overlaid
    :param img_batch: the batch of images in NCHW to visualize
    :param mask_batch_target: the batch of corresponding mask in NHW to visualize
    """
    img, mask_target = visualize_preprocess(img_batch, mask_batch_target)

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
    ax[0].imshow(img)
    ax[1].imshow(overlay_mask(mask_target, img))
    ax[0].set_title("Random Image in the Batch")
    ax[1].set_title("Overlaid Target Mask")
    plt.show()


def visualize_pred(
    img_batch: np.ndarray or torch.Tensor,
    mask_batch_pred: np.ndarray or torch.Tensor,
    mask_batch_target: np.ndarray or torch.Tensor,
):
    """
    Plot one image in the batch, the image overlaid with the ground truth mask,
    the image overlaid with the predicted mask, and the image overlaid with the predicted mask with
    wrong classifications mark red
    :param img_batch: the batch of images in NCHW to visualize
    :param mask_batch_pred: the batch of corresponding mask in NHW to visualize
    :param mask_batch_target: the batch of predicted masks (in NHW)
    :return:
    """
    img, mask_target, mask_pred = visualize_preprocess(img_batch, mask_batch_target, mask_batch_pred)
    mask_pred_overlay = overlay_mask(mask_pred, img)
    mask_target_overlay = overlay_mask(mask_target, img)
    mask_miss_classifications_overlay = mask_target_overlay.copy()
    # Mark wrong classifications as red
    mask_miss_classifications_overlay[mask_target != mask_pred, :] = torch.tensor([1., 0., 0.])

    _, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    ax[0, 0].imshow(img)
    ax[0, 1].imshow(mask_target_overlay)
    ax[1, 0].imshow(mask_pred_overlay)
    ax[1, 1].imshow(mask_miss_classifications_overlay)
    ax[0, 0].set_title("Random Image in the Batch")
    ax[0, 1].set_title("Overlaid Target Mask")
    ax[1, 0].set_title("Overlaid Prediction Mask")
    ax[1, 1].set_title("Overlaid Miss-Classification Mask")
    plt.show()


def overlay_mask(mask: np.ndarray, img: np.ndarray = None) -> np.ndarray:
    """
    Overlay mask to the given image with the color map specified in CDL_2013_clip_20170525181724_1012622514.tif.vat.dbf
    :param mask: Mask of the corresponding image in HW
    :param img: Image to be displayed in HWC
    :return: An image tensor with mask overlaid. The "Others" class is ignored
    """
    if img is not None:
        *_, c = img.shape
        assert c == 3
        return label2rgb(label=mask, image=img, alpha=0.3, colors=list(COLOR_MAP.values()), bg_label=0, kind="overlay")
    else:
        return label2rgb(label=mask, colors=list(COLOR_MAP.values()), bg_label=0)

