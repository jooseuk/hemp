import numpy as np
import random
import torch
import cv2
from torchvision import transforms as T

# ======================================
# Data Augmentation Functions for RIRA
# ======================================


def cutmix(input1: torch.Tensor,
           input2: torch.Tensor,
           label1: torch.Tensor,
           label2: torch.Tensor,
           alpha: float = 1.0):
    """
    Apply CutMix augmentation between two images and their labels.

    Args:
        input1 (Tensor): First input image tensor (C x H x W).
        input2 (Tensor): Second input image tensor to mix in.
        label1 (Tensor): One-hot or scalar label for input1.
        label2 (Tensor): One-hot or scalar label for input2.
        alpha (float): Parameter for the Beta distribution.

    Returns:
        mixed_input (Tensor): CutMix-augmented image.
        mixed_label (Tensor): Interpolated label.
    """
    # Sample mixing ratio from Beta distribution
    lam = np.random.beta(alpha, alpha)
    C, H, W = input1.shape

    # Determine patch size based on lambda
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Randomly choose center point of the patch
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # Calculate patch coordinates (ensure within image bounds)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Create mixed image by replacing the patch from input2
    mixed_input = input1.clone()
    mixed_input[:, bby1:bby2, bbx1:bbx2] = input2[:, bby1:bby2, bbx1:bbx2]

    # Adjust lambda to reflect exact area ratio
    area = (bbx2 - bbx1) * (bby2 - bby1)
    lam = area / (W * H)
    mixed_label = (1 - lam) * label1 + lam * label2

    return mixed_input, mixed_label


def mixup(input1: torch.Tensor,
          input2: torch.Tensor,
          label1: torch.Tensor,
          label2: torch.Tensor,
          alpha: float = 1.0):
    """
    Apply MixUp augmentation between two images and labels.

    Args:
        input1 (Tensor): First image (C x H x W).
        input2 (Tensor): Second image to mix.
        label1 (Tensor): Label for input1.
        label2 (Tensor): Label for input2.
        alpha (float): Beta distribution parameter.

    Returns:
        mixed_input (Tensor): Linear combination of inputs.
        mixed_label (Tensor): Linear combination of labels.
    """
    lam = np.random.beta(alpha, alpha)
    mixed_input = lam * input1 + (1 - lam) * input2
    mixed_label = lam * label1 + (1 - lam) * label2
    return mixed_input, mixed_label


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate a 2D array (mask) by a given angle.

    Args:
        image (ndarray): Input 2D mask or image.
        angle (float): Rotation angle in degrees.

    Returns:
        Rotated image with border reflection.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)


def grid_mask(image: torch.Tensor,
              num_grid: int = 3,
              fill_value: float = 0,
              rotate: int = 0) -> torch.Tensor:
    """
    Apply GridMask augmentation by zeroing out patches in a grid pattern.

    Args:
        image (Tensor): Input image (C x H x W).
        num_grid (int): Number of grid divisions.
        fill_value (float): Value to fill masked patches.
        rotate (int): Max rotation in degrees for the grid mask.

    Returns:
        Augmented image with grid mask applied.
    """
    C, h, w = image.shape
    # Randomize grid size variation
    grid_div = random.randint(num_grid, num_grid + 2)
    gh = h // grid_div
    gw = w // grid_div

    # Build binary mask
    mask = np.ones((h, w), dtype=np.float32)
    for i in range(grid_div):
        for j in range(grid_div):
            y1 = i * gh
            x1 = j * gw
            y2 = y1 + gh // 2
            x2 = x1 + gw // 2
            mask[y1:y2, x1:x2] = fill_value

    # Optionally rotate the mask
    if rotate:
        angle = random.uniform(-rotate, rotate)
        mask = rotate_image(mask, angle)

    mask = torch.from_numpy(mask).to(image.device)
    return image * mask

# Define a standard normal transform pipeline
normal_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.5),
    T.RandomAffine(degrees=(-45, 45),
                   shear=(-15, 15, -15, 15),
                   translate=(0.2, 0.2),
                   interpolation=T.InterpolationMode.BILINEAR,
                   fill=0),
    T.ColorJitter(brightness=0.1, contrast=0.1),
])

# List of augmentation functions and their sampling weights
aug_funcs = ['mixup', 'cutmix', 'grid_mask', 'normal', None]
weights   =  [1,       1,        1,           1,        4]

# Example batch augmentation loop
# Assume input_batch: Tensor[B x C x H x W], label_batch: Tensor[B x ...]
batch_size = input_batch.size(0)
perm       = torch.randperm(batch_size)
choices    = random.choices(aug_funcs, weights=weights, k=batch_size)

# Clone original inputs and labels
aug_input = input_batch.clone()
aug_label = label_batch.clone()

for i, aug in enumerate(choices):
    x1, y1 = input_batch[i], label_batch[i]
    x2, y2 = input_batch[perm[i]], label_batch[perm[i]]

    if aug == 'mixup':
        aug_input[i], aug_label[i] = mixup(x1, x2, y1, y2)
    elif aug == 'cutmix':
        aug_input[i], aug_label[i] = cutmix(x1, x2, y1, y2)
    elif aug == 'grid_mask':
        aug_input[i] = grid_mask(x1, num_grid=2, fill_value=0, rotate=90)
    elif aug == 'normal':
        aug_input[i] = normal_transform(x1)
    # if aug is None, keep original sample

# Finally, apply normalization (replace MEAN and STD with dataset stats)
normalize = T.Normalize(mean=MEAN, std=STD)
aug_input = normalize(aug_input)
