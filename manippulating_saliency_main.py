# USAGE EXEMPLE: python .\manippulating_saliency_main.py .\data\debug\easy_apple_small.jpg .\data\debug\easy_apple_small_mask.jpg 0.1

# Libraries imports
import numpy as np
import matplotlib.pyplot as plt
import cv2, skimage, os, sys

# Local imports
from src import saliancy_map as sm
from src import utils
from src import optimization as opt
from src import database as db
from src.optimization import get_pyramids, reconstruct

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
def manipulate_saliency(input_image, R, delta_s, max_iteration=10, patch_size=7, learning_rate=0.1):
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
    tau_positive = 0.5
    tau_negative = 0.5

    print("\ninitalizing variables")

    # Initialize the interational images buffers (Try to keep them as [0,255] images)
    J = np.array([np.zeros_like(input_image) for _ in range(2)])
    J[0] = cv2.cvtColor(input_image, cv2.COLOR_RGB2Lab)

    # Initialize the saliency maps S_J
    print(" - computing saliency map...")
    S_J = compute_saliency_map(J[0])
    print(" - Done.")

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
    print(" - Current saliency contrast:", phi(S_J, R), "Objective:", delta_s)
    print("\nBegin Saliency Manipulation:")

    # while compute_criterion(S_J, R, delta_s) > EPSILON:
    for i in  range(max_iteration):
        print(f"Iteration {i}")

        # DB update
        print(f" - computing DB with tau+ = {tau_positive}, tau- = {tau_negative}...")
        D_positive, D_negative, D_pos_mask, D_neg_mask = compute_database(tau_positive, tau_negative, J[0], S_J)
        print(f" - Done, DB+ size: {D_positive.shape[0]}, DB- size: {D_negative.shape[0]}")
        
        # Construct and display the database's images
        # I_D_positive, I_D_negative = db.compute_image_database(J[0], D_positive, D_negative)

        # utils.display_images([S_J, I_D_positive, I_D_negative])

        # update J to minimize the energy function
        print(" - Minimizing function...")
        J[1] = minimize_J(J[0], mask_image, D_positive, D_negative, D_pos_mask, D_neg_mask, patch_size)
        print(" - Done.")

        # Compute new saliency map
        print(" - computing new saliency map...")
        S_J = compute_saliency_map(J[1])
        print(" - Done.")

        # Compute criterions
        criterion = compute_criterion(S_J, R, delta_s)
        print(" - Iteration's saliency contrast:", phi(S_J, R), "Objective:", delta_s)

        # Check if convergence is reached by tau's
        tau_diff = learning_rate * criterion
        if tau_diff < EPSILON:
            print(" - Tau's convergence reached.")
            break
        
        # Check if convergence is reached by delta_s
        if criterion < EPSILON:
            print(" - Criterion convergence reached.")
            break

        # Update tau +/-
        tau_positive, tau_negative = update_taus(tau_positive, tau_negative, criterion, learning_rate)

        # switch the buffers (only affect the references so no copy is made)
        temp = J[0].copy()
        J[0] = J[1]
        J[1] = temp


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
    foreground = S_J[np.where(R > 0)]   # gathers the point inside region

    percentile = np.percentile(foreground, 1 - thresh)
    f_mean = foreground[np.where(foreground > percentile)].mean()
    
    background = S_J[np.where(R == 0)]   # gathers the point outsid region

    percentile = np.percentile(background, 1 - thresh)
    b_mean = background[np.where(background > percentile)].mean()
    
    return f_mean - b_mean

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

def update_taus(tau_positive, tau_negative, criterion, learning_rate):
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
    adjustment = criterion * learning_rate
    tau_positive += adjustment
    tau_negative -= adjustment

    return tau_positive, tau_negative
        

# Entry point
if __name__ == "__main__":

    assert len(sys.argv) == 4, "Usage: python manipulating_saliency_main.py <image_path> <mask_path> <delta_s>"
    image_path = sys.argv[1]
    mask_path = sys.argv[2]
    delta_s = float(sys.argv[3])

    
    # Check that the path corresponds to existing files
    if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Path {image_path} does not exist.")

    # Read the image
    input_image = cv2.imread(image_path)
    print("\n - Image size:", input_image.shape)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Read Mask    
    mask_image = cv2.imread(mask_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)[:,:]
    
    mask_image = cv2.resize(mask_image, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    assert input_image.shape[:2] == mask_image.shape[:2], "Image and mask must match in size"

    _, mask_image = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)

    # # Re binearize the mask
    # non_zeros = np.where(mask_image != 0)
    # mask_image[non_zeros] = 1

    # Compute the pyramids
    pyramids = get_pyramids(input_image, 3)
    mask_pyramids = get_pyramids(mask_image, 3)
    
    # TEMP
    input_image = pyramids[0][1]
    
    mask_image = mask_pyramids[0][1]

    # Call the main function
    salient_image = manipulate_saliency(input_image, mask_image, delta_s)
    
    # Display the original image and the saliency map
    utils.display_images([input_image, salient_image, mask_image], ["intput", "modified", "mask"])
    # save the orinal image
    # save_image(image, folder_path + "original_image.jpg")

    plt.show()
    