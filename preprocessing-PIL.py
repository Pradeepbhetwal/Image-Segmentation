import numpy as np
from PIL import Image

def preprocess_segmentation(image_path, mask_path, modality="MRI",
                            target_size=(256, 256), window=(-1000, 400)):
    """
    Preprocess an image and mask for segmentation training.
    
    Args:
        image_path (str): Path to input image (MRI or CT).
        mask_path  (str): Path to corresponding mask image.
        modality   (str): 'MRI' or 'CT'.
        target_size(tuple): Desired output size (H, W).
        window     (tuple): CT window (min_HU, max_HU).

    Returns:
        image (np.ndarray): Standardized image [H, W].
        mask  (np.ndarray): Integer mask [H, W].
    """

    # --- Load image ---
    image = np.array(Image.open(image_path).convert("F"), dtype=np.float32)  # grayscale
    mask  = np.array(Image.open(mask_path), dtype=np.int32)  # keep integer labels

    # --- Resize ---
    image = np.array(Image.fromarray(image).resize(target_size, resample=Image.BILINEAR))
    mask  = np.array(Image.fromarray(mask).resize(target_size, resample=Image.NEAREST))  # no interpolation

    # --- Modality-specific normalization ---
    if modality.upper() == "MRI":
        mean, std = np.mean(image), np.std(image)
        image = (image - mean) / (std + 1e-8)  # Z-score normalization

    elif modality.upper() == "CT":
        min_HU, max_HU = window
        image = np.clip(image, min_HU, max_HU)  # clip to HU window
        image = (image - min_HU) / (max_HU - min_HU)  # scale to [0,1]

    else:
        raise ValueError("Modality must be 'MRI' or 'CT'.")

    return image, mask
