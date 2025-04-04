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

def get_picture_path(image_info):
    """
    Given the coco image info, retreive the path in our project structure
    """
    file_name = image_info['file_name']
    img_path = os.path.join(IMG_DIR, file_name)
    return img_path

if __name__ == "__main__":
    
    annFile = os.path.join(ANNOT_DIR, "instances_val2017.json")
    coco = COCO(annFile)
    
    category_ids = coco.getCatIds()
    cats = coco.loadCats(category_ids)
    cat_names = [cat['name'] for cat in cats]
    
    # Get annotations
    imgIds = coco.getImgIds()
    img = coco.loadImgs(imgIds[100])[0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img['id'], catIds=category_ids, iscrowd=None))
    
    # Display image with annotations
    img_path = get_picture_path(img)
    image = Image.open(img_path)   
        
    mask = coco.annToMask(anns[0])
    # add the mask to the image
    image_out = np.array(image)
    mask = np.array(mask)
    image_out[mask > 0] = [255, 0, 0]  # Red color for the mask

    plt.imshow(image_out)
    plt.axis('off')
    plt.show()
    
    # save the picture and mask
    output_img_path = os.path.join(OUTPUT_DIR, img['file_name'])
    output_mask_path = os.path.join(OUTPUT_DIR, img['file_name'].replace('.jpg', '_mask.jpg'))

    # Set all the mask pixel to white
    mask[mask > 0] = 255
    mask = Image.fromarray(mask.astype(np.uint8))
    
    # Save the image and mask
    image = Image.fromarray(np.array(image).astype(np.uint8))
    image.save(output_img_path)
    mask.save(output_mask_path)
    print(f"Saved image to {output_img_path}")