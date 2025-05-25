# Contains different function to return

import cv2, skimage
import numpy as np
from pathlib import Path
import sys
import torch
# from src.tempsal_wrapper import compute_saliency_map


CV_SALIENCY_COARSE = cv2.saliency.StaticSaliencySpectralResidual().create()
CV_SALIENCY_FINE = cv2.saliency.StaticSaliencyFineGrained().create()

def tempsal_saliency(image):
    saliency = compute_saliency_map(image).mean(axis=2)
    # Scale the map so that max value = 1
    saliency /= saliency.max()

    # import matplotlib.pyplot as plt
    # plt.plot(saliency[75,:])

    # Linearize the values
    # saliency = saliency / (saliency + np.median(saliency))

    # print("median of the new saliency:", np.median(saliency))
    # import matplotlib.pyplot as plt
    # plt.plot(saliency[75,:])
    # plt.show()


    return saliency

def custom_saliency(input_image):
    """
    Computes the saliency map using the method given in the paper

    This function is a placeholder and currently does not implement any functionality.
    Future implementations should include the logic to compute the saliency map.

    Returns:
        None
    """
    lab_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2Lab)
    L, A, B = cv2.split(lab_image)

    # Compute the color contrast
    chroma = np.sqrt(A.astype(np.float32) ** 2 + B.astype(np.float32) ** 2)

    C_median = np.median(chroma)    
    L_median = np.percentile(L, 33)     # Favoring lighntess over darkness

    chroma = np.abs(chroma - C_median)
    L = np.abs(L - L_median)

    # Nomalize for comparable values
    C_norm = (chroma - chroma.min()) / (chroma.max() - chroma.min()) 
    L_norm = (L - L.min()) / (L.max() - L.min()) 

    s_map = np.sqrt(C_norm ** 2 + L_norm ** 3)

    return s_map

def course_saliancy(input_image):
    """
    Computes the saliency map using the formula given in the course, which is S(x,y)=||I_{mean} - I_{pixel}(x,y)||

    This function is implementing a very basic saliency map computation. It computes the saliency map using the formula given in the course and it will be mostly used in early steps of the project. 

    Args:
        input_image: The input image for which the saliency map is to be computed

    Returns:
        output_image: The saliency map for the input image
    """

    # We want to apply the formula : ||I_{mean} - I_{pixel}(x,y)|| for every pixel of the given image
    mean = np.mean(input_image, axis=(0,1))
    s_map = np.linalg.norm(mean-input_image, axis=2)
    return s_map


def opencv_saliency(input_image):
    """
    Computes the saliency map for a given image using the static opcv methode
    Returns:
        None
    """
    _, saliency = CV_SALIENCY_COARSE.computeSaliency(input_image)
    saliency /= saliency.max()
    assert not np.any(saliency < 0)
    return saliency

def apply_saliancy(input_image, saliancy_map, alpha):
    """
    Tweaks the input image based on the saliancy map.

    Args:
        input_image: The input image to be tweaked
        saliancy_map: The saliancy map based on which the input image is to be tweaked
        alpha: The constant to be multiplied with the saliancy map

    Returns:
        output_image: The tweaked image
    """

    # We want to apply the formula : I_{pixel}(x,y) = I_{pixel}(x,y) + S(x,y) * alpha
    # where alpha is a constant

    factor = saliancy_map / saliancy_map.max()

    output_image = input_image * np.power(factor, alpha)

    # output_image = input_image + saliancy_map * alpha
    output_image = np.clip(output_image, 0, 255).astype(np.uint8)
    

    return output_image

def apply_range_of_saliancy(input_image, saliancy_map, alphas):
    output_images = []
# 
    for alpha in alphas:
        modified_image = apply_saliancy(input_image, saliancy_map, alpha)
        output_images.append(modified_image)
        
    return output_images