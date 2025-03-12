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

def display_images(image_list):
    """
    Displays a whole list of images next to each other

    Args:
        imgae_list: A list of images to display

    Returns:
        None
    """

    # Compute number of rows
    x = len(image_list)//3


    # Create the figure
    fig, axs = plt.subplots(x, 3)

    # Fill the figure with the images
    if x == 1:
        for j in range(len(image_list)):
            axs[j].imshow(image_list[j])
            axs[j].axis('off')

    else:
        for i in range(x):
            for j in range(3):
                axs[i, j].imshow(image_list[i*3+j])
                axs[i, j].axis('off')
  
    # display
    plt.show()

    