# Libraries imports
import numpy as np
import matplotlib.pyplot as plt
import cv2, skimage, os, sys

# Local imports
from src import saliancy_map as sm
from src import utils

# Constants
DATA_PATH = "data/"
DEBUG_IMAGES_PATH = DATA_PATH + "debug/"
DEFAULT_IMAGE = DEBUG_IMAGES_PATH + "easy_apple.jpg"

# Main function
def main(input_image):
    """
    Main function, for now it's debug
    """

    # Compute the saliancy map
    saliancy_map = sm.course_saliancy(input_image)

    # Apply it to the image
    modified_images = sm.apply_range_of_saliancy(input_image, saliancy_map, [0.1, 0.2, 0.5, 0.8, 1, 2, 4])
    # modified_image = sm.apply_saliancy(input_image, saliancy_map, 0.5)
    
    # Display
    # utils.display_images([input_image, saliancy_map, modified_image])
    utils.display_images([input_image, saliancy_map] + modified_images)
    # utils.display_image(input_image, saliancy_map)


# Entry point
if __name__ == "__main__":
    # Check the number of arguments
    if len(sys.argv) == 2:
        image_path = sys.argv[1]
    else:
        image_path = DEFAULT_IMAGE

    # Read the image
    input_image = cv2.imread(image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Call the main function
    main(input_image)

    