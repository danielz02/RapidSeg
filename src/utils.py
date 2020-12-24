import torch
import numpy as np
import matplotlib.pyplot as plt
from src.params import COLOR_MAP
from skimage.color import label2rgb


def normalize(arr: np.ndarray):
    arr_min, arr_max = np.min(arr), np.max(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def visualize_preprocess(
    img_batch: np.ndarray or torch.Tensor,
    mask_batch_target: np.ndarray or torch.Tensor,
    mask_batch_pred: np.ndarray or torch.Tensor = None
):
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
    img, mask_target = visualize_preprocess(img_batch, mask_batch_target)

    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 8))
    ax[0].imshow(img)
    ax[1].imshow(overlay_mask(mask_target, img))
    ax[0].set_title("Random Image in the Batch")
    ax[1].set_title("Overlaid Target Mask")


def visualize_pred(
    img_batch: np.ndarray or torch.Tensor,
    mask_batch_pred: np.ndarray or torch.Tensor,
    mask_batch_target: np.ndarray or torch.Tensor,
):
    img, mask_target, mask_pred = visualize_preprocess(img_batch, mask_batch_target, mask_batch_pred)
    mask_pred_overlay = overlay_mask(mask_pred, img)
    mask_target_overlay = overlay_mask(mask_target, img)
    mask_miss_classifications_overlay = mask_target_overlay.copy()
    mask_miss_classifications_overlay[mask_target != mask_pred, :] = torch.tensor([1., 0., 0.])

    print(mask_pred_overlay.shape)

    _, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    ax[0, 0].imshow(img)
    ax[0, 1].imshow(mask_target_overlay)
    ax[1, 0].imshow(mask_pred_overlay)
    ax[1, 1].imshow(mask_miss_classifications_overlay)
    ax[0, 0].set_title("Random Image in the Batch")
    ax[0, 1].set_title("Overlaid Target Mask")
    ax[1, 0].set_title("Overlaid Prediction Mask")
    ax[1, 1].set_title("Overlaid Miss-Classification Mask")


def overlay_mask(mask: np.ndarray, img: np.ndarray = None) -> np.ndarray:
    if img is not None:
        *_, c = img.shape
        assert c == 3
        return label2rgb(label=mask, image=img, alpha=0.3, colors=COLOR_MAP.values(), bg_label=0)
    else:
        return label2rgb(label=mask, colors=COLOR_MAP.values(), bg_label=0)

