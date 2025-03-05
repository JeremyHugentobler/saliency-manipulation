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
    saliancy_map = sm.course_saliancy(input_image)
    utils.display_image(saliancy_map, input_image)


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

    