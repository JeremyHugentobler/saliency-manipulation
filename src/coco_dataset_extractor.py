# To be executed from the root directory

ANNOT_DIR = "data/coco_input/annotations/"
IMG_DIR = "data/coco_input/val2017/"
OUTPUT_DIR = "data/coco_output/"

import os
import json
from pycocotools.coco import COCO

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Button

def get_picture_path(image_info):
    """
    Given the coco image info, retreive the path in our project structure
    """
    file_name = image_info['file_name']
    img_path = os.path.join(IMG_DIR, file_name)
    return img_path

def save_results(event):
    """
    Save the current image and mask to the specified paths.
    """
    global image, mask, output_img_path, output_mask_path
    image.save(output_img_path)
    mask.save(output_mask_path)
    print(f"Saved image to {output_img_path}")
    print(f"Saved mask to {output_mask_path}")

def update_mask(ax, image, anns, mask_index):
    """
    Update the displayed mask on the image.
    """
    mask = coco.annToMask(anns[mask_index])
    image_out = np.array(image)
    mask = np.array(mask)
    image_out[mask > 0] = [255, 0, 0]  # Red color for the mask

    ax.clear()
    ax.imshow(image_out)
    ax.axis('off')
    plt.draw()
    return mask

if __name__ == "__main__":
    
    annFile = os.path.join(ANNOT_DIR, "instances_val2017.json")
    coco = COCO(annFile)
    
    category_ids = coco.getCatIds()
    cats = coco.loadCats(category_ids)
    cat_names = [cat['name'] for cat in cats]
    
    # Get annotations
    imgIds = coco.getImgIds()
    img_index = 100  # Start with image at index 100
    img = coco.loadImgs(imgIds[img_index])[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img['id'], catIds=category_ids, iscrowd=None))
    
    # Display image with annotations
    img_path = get_picture_path(img)
    image = Image.open(img_path)
    mask_index = 0  # Start with the first mask

    fig, ax = plt.subplots()
    mask = update_mask(ax, image, anns, mask_index)

    # Add a "Save" button
    ax_button_save = plt.axes([0.8, 0.01, 0.1, 0.05])  # Position of the "Save" button
    button_save = Button(ax_button_save, 'Save')

    # Add a "Next Mask" button
    ax_button_next = plt.axes([0.65, 0.01, 0.1, 0.05])  # Position of the "Next Mask" button
    button_next = Button(ax_button_next, 'Next Mask')

    # Add a "Next Image" button
    ax_button_next_img = plt.axes([0.5, 0.01, 0.1, 0.05])  # Position of the "Next Image" button
    button_next_img = Button(ax_button_next_img, 'Next Image')

    # Set all the mask pixel to white
    mask[mask > 0] = 255
    mask = Image.fromarray(mask.astype(np.uint8))
    image = Image.fromarray(np.array(image).astype(np.uint8))

    # Global save paths
    output_img_path = os.path.join(OUTPUT_DIR, img['file_name'])
    output_mask_path = os.path.join(OUTPUT_DIR, img['file_name'].replace('.jpg', '_mask.jpg'))

    # Define the "Save" button click event
    button_save.on_clicked(save_results)

    # Define the "Next Mask" button click event
    def next_mask(event):
        global mask_index, mask
        mask_index = (mask_index + 1) % len(anns)  # Cycle through masks
        mask = update_mask(ax, image, anns, mask_index)
        
        # Make sure to update the mask for saving
        temp_mask = np.array(mask)
        temp_mask[temp_mask > 0] = 255
        mask = Image.fromarray(temp_mask.astype(np.uint8))

    # Define the "Next Image" button click event
    def next_image(event):
        global img_index, mask_index, img, anns, image, mask, output_img_path, output_mask_path
        
        # Move to next image
        img_index = (img_index + 1) % len(imgIds)
        img = coco.loadImgs(imgIds[img_index])[0]
        
        # Reset mask index to 0
        mask_index = 0
        
        # Load new image
        img_path = get_picture_path(img)
        image = Image.open(img_path)
        
        # Get annotations for the new image
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img['id'], catIds=category_ids, iscrowd=None))
        
        # Update display with first mask
        mask = update_mask(ax, image, anns, mask_index)
        
        # Update mask for saving
        temp_mask = np.array(mask)
        temp_mask[temp_mask > 0] = 255
        mask = Image.fromarray(temp_mask.astype(np.uint8))
        image = Image.fromarray(np.array(image).astype(np.uint8))
        
        # Update save paths
        output_img_path = os.path.join(OUTPUT_DIR, img['file_name'])
        output_mask_path = os.path.join(OUTPUT_DIR, img['file_name'].replace('.jpg', '_mask.jpg'))

    button_next.on_clicked(next_mask)
    button_next_img.on_clicked(next_image)

    plt.show()