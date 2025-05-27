import numpy as np
import matplotlib.pyplot as plt
import cv2, skimage, os, sys

VERBOSE = False
DISP_OUT = True

def header_print(title):
    """Function to print a title with # around it
    Args:
        title (str): What you want to be printed
    """
    
    print('#'*(len(title) + 8))
    print('#'*4, title, '#' * 4)
    print('#'*(len(title) + 8))

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

def display_images(images, titles=[], imperrow=3):
    """
    Displays a whole list of images next to each other

    Args:
        imgae_list: A list of images to display
        titles: A list of the corrsponding title (optional)

    Returns:
        None
    """

    # Compute number of rows
    x = len(images) // imperrow + 1

    # Fill the figure with the images
    plt.figure(figsize=(30,20))
    for i in range(x):
        l = len(images)%imperrow if i+1 == x else imperrow
        for j in range(l):
            idx = i*imperrow + j
            plt.subplot(x+1, imperrow, idx+1)
            plt.imshow(images[idx], cmap='coolwarm')
            plt.axis('off')
            if len(titles) != 0:
                plt.title(titles[idx]) 
  
    # display
    plt.show()

    