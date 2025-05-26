from pathlib import Path
from PIL import Image
# Constants
REF_PATH = Path(".\data\saliency_shift")
MASK_PATH = REF_PATH / "masks"


# The goal is to rescale to mask to the size of the image
# The mask is a binary image, so we can use the PIL library to open it
# /!\ The output should be BINARY


extensions = [".jpg", ".png", ".jpeg"]
in_images_path = []
for ext in extensions:
    # get all the images, but exclude subdirectories
    
    in_images_path.extend(REF_PATH.glob(f"**/*_in*{ext}"))
# Remove the subdirectories
in_images_path = [img for img in in_images_path if img.parent == REF_PATH]


masks_path = []
for ext in extensions:
    masks_path.extend(MASK_PATH.glob(f"**/*{ext}"))
    
    
for img_path in in_images_path:
    # Get the corresponding mask
    mask_path = None
    out = img_path.stem.replace("_in", "")
    for m_path in masks_path:
        if m_path.stem == out + "_mask":
            mask_path = m_path
            break
    if mask_path is None:
        raise ValueError(f"Mask for {img_path} not found.")
    
    # Open the image and mask
    img = Image.open(img_path)
    mask = Image.open(mask_path)
    # Resize the mask to the size of the image
    mask = mask.resize(img.size, Image.NEAREST)
    # treshold the mask to make it binary
    # Save the mask at its original location
    mask.save(mask_path)
    