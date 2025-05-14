# USAGE EXEMPLE: python .\manippulating_saliency_main.py .\data\debug\easy_apple_small.jpg .\data\debug\easy_apple_mask_small.jpg 0.1

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


############################
#    Modular Definitions   # 
############################
compute_saliency_map = sm.tempsal_saliency
minimize_J = opt.minimize_J_global_poisson
compute_database = db.compute_location_database



# Main function
def manipulate_saliency(input_image, R, delta_s, max_iteration=10, patch_size=7):
    """
    This is the main function that will implement the saliency manipulation algorithm
    as described in the paper. The input image will see its region of interest R defined by
    the mask modified such that the resulting image will have a saliency contrast between the
    region R and the rest of the image to be delta_s.

    Args:
        input_image: The input image
        R: The mask defining the region of interest
        delta_s: The saliency contrast to be achieved
        max_iteration: max number of iterations 
        patch_size: the size used by the patch-match function
    Returns:
        None
    """

    ################################
    # Initialize all the variables #
    ################################

    # Initialize tau +/-
    tau_positive = 0
    tau_negative = 1
    prev_tau_positive = tau_positive
    prev_tau_negative = tau_negative

    # Initialize the interational images buffers (Try to keep them as [0,255] images)
    J = np.array([np.zeros_like(input_image) for _ in range(2)])
    J[0] = cv2.cvtColor(input_image, cv2.COLOR_RGB2Lab)

    # Initialize the saliency maps S_J
    S_J = np.zeros(input_image.shape[0:2])

    # Erode the mask so that only patches within are modified
    
    radius = patch_size // 2
    kernel = np.ones((radius, radius))
    mask_image = cv2.erode(R, kernel)

    # Initialize the Database I_D +/-
    # I_D_positive, I_D_negative = [np.zeros_like(input_image) for _ in range(2)]
    

    ##############################
    # Iteration of the algorithm #
    ##############################

    # TODO: make the coarse-to-fine iterations
    print("\nBegin Saliency Manipulation:")

    # while compute_criterion(S_J, R, delta_s) > EPSILON:
    for i in  range(max_iteration):
        print(f"Iteration {i}")
        # update the saliency map
        print(" - computing Saliency...")
        S_J = compute_saliency_map(J[0])
        print(" - Done.")

        # DB update
        print(" - computing DB...")
        D_positive, D_negative = compute_database(tau_positive, tau_negative, J[0], S_J)
        print(f" - Done, DB+ size: {D_positive.shape[0]}, DB- size: {D_negative.shape[0]}")
        
        # Construct and display the database's images
        I_D_positive, I_D_negative = db.compute_image_database(J[0], D_positive, D_negative)

        # utils.display_images([S_J, I_D_positive, I_D_negative])

        # update J to minimize the energy function
        print(" - Minimizing function...")
        J[1] = minimize_J(J[0], mask_image, D_positive, D_negative, patch_size)
        print(" - Done.")
        # Update tau +/-
        tau_positive, tau_negative = update_taus(tau_positive, tau_negative, S_J, mask_image, delta_s)

        # switch the buffers (only affect the references so no copy is made)
        temp = J[0].copy()
        J[0] = J[1]
        J[1] = temp

        # Check if convergence is reached by tau's
        tau_diff = abs(tau_positive - prev_tau_positive) + abs(tau_negative - prev_tau_negative)
        prev_tau_positive, prev_tau_negative = tau_positive, tau_negative
        if tau_diff < EPSILON:
            break
        
        # Check if convergence is reached by delta_s
        if compute_criterion(S_J, mask_image, delta_s) > EPSILON:
            break

        # print("\033[A\033[K\033[A\033[K\033[A\033[K\033[A\033[K\033[A\033[K\033[A\033[K\033[A\033[K", end="")

    print("Done")
    return cv2.cvtColor(J[0], cv2.COLOR_Lab2RGB)

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
    thresh = 0.2
    foreground = S_J.copy()
    foreground[R == 0] = 0  # Inside target region
    # Magic to retrive top 20%
    min, max = np.min(foreground), np.max(foreground)
    x,y = np.where(foreground < min + (1 - thresh) * (max - min))
    foreground[x,y] = 0

    background = S_J.copy()
    background[R > 0] = 0  # Outside target region
    # Magic to retrive top 20% by seting to 0 the other 80%
    min, max = np.min(background), np.max(background)
    x,y = np.where(background < min + (1 - thresh) * (max - min))
    background[x,y] = 0

    return foreground.mean() - background.mean()

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
    return np.abs(phi(S_J, R) - delta_s)

def update_taus(tau_positive, tau_negative, S_J, R, delta_s, learning_rate=0.1):
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
    # Adjust thresholds based on difference from desired contrast
    adjustment = np.abs(phi(S_J, R) - delta_s) * learning_rate
    tau_positive += adjustment
    tau_negative -= adjustment

    return tau_positive, tau_negative
        

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
    print("\n - Image size:", input_image.shape)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    
    mask_image = cv2.imread(mask_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)[:,:]

    # Call the main function
    salient_image = manipulate_saliency(input_image, mask_image, delta_s)
    
    # Display the original image and the saliency map
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(input_image)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(salient_image, cmap='hot')
    plt.axis('off')

    # save the orinal image
    # save_image(image, folder_path + "original_image.jpg")

    plt.show()
    