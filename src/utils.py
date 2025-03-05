import numpy as np
import matplotlib.pyplot as plt
import cv2, skimage, os, sys


def display_image(result_image, input_image):
    """
    Displays the result image next to the default image

    Args:
        result_image: The image twith modification
        input_image: The original image 

    Returns:
        None
    """

    plt.imshow(np.hstack((input_image, result_image)))
    # Change title
    plt.title("Comparison between the original and the modified image")
    # Disable axis
    plt.axis('off')
    # display
    plt.show()
    