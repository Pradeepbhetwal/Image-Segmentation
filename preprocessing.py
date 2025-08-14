from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, NormalizeIntensityd,
    Resized, EnsureTyped
)
from monai.data import Dataset, DataLoader

# --- Define data list (images + masks) ---
train_files = [
    {"image": "images/CT_001.nii.gz", "label": "masks/CT_001_mask.nii.gz"},
    {"image": "images/CT_002.nii.gz", "label": "masks/CT_002_mask.nii.gz"},
    # ...
]

# --- Define transforms ---
train_transforms = [
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),  # C,H,W or C,H,W,D
    Orientationd(keys=["image", "label"], axcodes="RAS"),  # standard orientation
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    
    # CT normalization example:
    ScaleIntensityRanged(
        keys=["image"],
        a_min=-1000, a_max=400,  # HU window
        b_min=0.0, b_max=1.0,
        clip=True
    ),
    
    # MRI alternative:
    # NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    
    Resized(keys=["image", "label"], spatial_size=(128, 128, 64),
            mode=("trilinear", "nearest")),
    
    EnsureTyped(keys=["image", "label"]),
]

# --- Create dataset ---
train_ds = Dataset(data=train_files, transform=train_transforms)

# --- DataLoader ---
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
