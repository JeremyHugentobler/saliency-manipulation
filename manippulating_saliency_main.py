# Libraries imports
import numpy as np
import matplotlib.pyplot as plt
import cv2, skimage, os, sys

# Local imports
from src import saliancy_map as sm
from src import utils
from src import optimization as opt
from src import database as db

# Constants

# DATA_PATH = "data/"
# DEBUG_IMAGES_PATH = DATA_PATH + "debug/"
# DEFAULT_IMAGE = DEBUG_IMAGES_PATH + "easy_apple.jpg"

EPSILON = 1e-3
compute_saliency_map = sm.paper_saliancy


# Main function
def manipulate_saliency(input_image, R, delta_s):
    """
    This is the main function that will implement the saliency manipulation algorithm
    as described in the paper. The input image will see its region of interest R defined by
    the mask modified such that the resulting image will have a saliency contrast between the
    region R and the rest of the image to be delta_s.

    Args:
        input_image: The input image
        R: The mask defining the region of interest
        delta_s: The saliency contrast to be achieved
    Returns:
        None
    """

    ################################
    # Initialize all the variables #
    ################################

    # Initialize tau +/-
    tau_plus = 0
    tau_minus = 1 #TODO is it right ?
    prev_tau_plus = tau_plus
    prev_tau_minus = tau_minus

    # Initialize the interational images buffers
    J = [np.arraylike(input_image) for _ in range(2)]

    # Initialize the saliency maps S_J
    S_J = np.zeros_like(input_image)

    # Initialize the Database I_D +/-
    I_D_plus, I_D_minus = [np.arraylike(input_image) for _ in range(2)]
    

    ##############################
    # Iteration of the algorithm #
    ##############################

    # TODO: make the coarse-to-fine iterations

    while compute_criterion(S_J, R, delta_s) > EPSILON:
        # update the saliency map
        S_J = compute_saliency_map(J[0], patch_size=5)

        # DB update
        I_D_plus, I_D_minus = db.compute_database(tau_plus, tau_minus, J[0], S_J)

        # Update tau +/-
        tau_plus, tau_minus = update_taus(tau_plus, tau_minus, S_J, R, delta_s)
        

        # update J to minimize the energy function
        J[1] = opt.minimize_J(J[0], I_D_plus, I_D_minus, R)

        # switch the buffers (only affect the references so no copy is made)
        temp = J[0]
        J[0] = J[1]
        J[1] = temp

        # Check if convergence is reached by tau's
        tau_diff = abs(tau_plus - prev_tau_plus) + abs(tau_minus - prev_tau_minus)
        if tau_diff < EPSILON:
            break

def phi(S_J, R):
    """
    Compute the saliency contrast between the region of interest R and the rest of the image
        phi(S_J, R) = mean(top(S_J[R])) - mean(top(S_J[not R]))
    where: top(x) keeps only the top 20% of the values of x

    Args:
        S_J: The saliency map of the current image J
        R: The region of interest
    Returns:
        The saliency contrast between R and the rest of the image
    """
    # TODO
    pass    

def compute_criterion(S_J, R, delta_s):
    """
    Compute the convergence criterion of the algorithm given as 
        ||phi(S_J, R) - delta_s||
    Args:
        S_J: The saliency map of the current image J
        R: The region of interest
        delta_s: The desired saliency contrast
    Returns: 
        The value of the criterion
    """
    # TODO
    pass

def update_taus(tau_plus, tau_minus, S_J, R, delta_s):
    """
    Update the tau +/- values based on the saliency maps and the target saliency contrast

    Args:
        tau_plus: The current value of tau_plus
        tau_minus: The current value of tau_minus
        S_J: The saliency map of the current image J
        R: The region of interest
        delta_s: The desired saliency contrast
    Returns:
        The updated values of tau_plus and tau_minus
    """
    # TODO
    pass
        

# Entry point
if __name__ == "__main__":
    # Check the number of arguments
    # if len(sys.argv) == 2:
    #     image_path = sys.argv[1]
    # else:
    #     image_path = DEFAULT_IMAGE

    assert len(sys.argv) == 4, "Usage: python manipulating_saliency_main.py <image_path> <mask_path> <delta_s>"
    image_path = sys.argv[1]
    mask_path = sys.argv[2]
    delta_s = float(sys.argv[3])

    # Read the image
    input_image = cv2.imread(image_path)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.imread(mask_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    # Call the main function
    manipulate_saliency(input_image, mask_image, delta_s)

    