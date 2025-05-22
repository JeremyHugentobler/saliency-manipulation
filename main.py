# USAGE EXEMPLE: python .\main.py .\data\debug\easy_apple_small.jpg .\data\debug\easy_apple_small_mask.jpg 0.1

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

EPSILON = 1e-4


############################
#    Modular Definitions   # 
############################
compute_saliency_map = sm.opencv_saliency
minimize_J = opt.minimize_J_global_poisson
compute_database = db.compute_location_database



def manipulate_saliency(input_image, R, delta_s, max_iteration=10, patch_size=3, learning_rate=0.1, convert=True):
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

    print("\ninitalizing variables")

    # Initialize the interational images buffers (Try to keep them as [0,255] images)
    J = np.array([np.zeros_like(input_image) for _ in range(2)])
    J[0] = input_image
    
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
        
        # Convert image into Lab color space
        if convert:
            J[0] = cv2.cvtColor(J[0], cv2.COLOR_RGB2Lab)

        J[1] = minimize_J(J[0], mask_image, D_positive, D_negative, D_pos_mask, D_neg_mask, patch_size)
        print(" - Done.")

        # Convert back into RGB
        if convert:
            J[0] = cv2.cvtColor(J[0], cv2.COLOR_Lab2RGB)
            J[1] = cv2.cvtColor(J[1], cv2.COLOR_Lab2RGB)

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
    return J[0], tau_positive, tau_negative

def image_update_only(input_image, R, iterations, tau_positive, tau_negative, patch_size=7):
    """Called for image at finer scales, it's basically only the second part of 'manipulate_saliency' and refered as 'image update' in the paper.

    Args:
        input_image: The image (not the coarsest one) on which to apply the image update
        R: mask defining the region of interest
        iterations: number of iterations to run (5 to 20 mentionned in the paper)
        tau_positive : tau found in the manipulate_saliency function
        tau_negative : tau found in the manipulate_saliency function
        patch_size (int, optional): The patch size. Defaults to 7.
    """    
    
    # Input in RGB, transform to Lab
    image = input_image.copy()
        
    for i in range(iterations):
        
        # Get the saliency map
        s_map = compute_saliency_map(image)
        # Get the databases
        D_positive, D_negative, D_pos_mask, D_neg_mask = compute_database(tau_positive, tau_negative, image, s_map)
        
        # Convert in Lab
        image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        # Compute the image update
        image = minimize_J(image, R, D_positive, D_negative, D_pos_mask, D_neg_mask, patch_size)
        # Convert back in RGB
        image = cv2.cvtColor(image, cv2.COLOR_Lab2RGB)

    return image

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

    if len(sys.argv) == 4:
        image_path = sys.argv[1]
        mask_path = sys.argv[2]
        delta_s = float(sys.argv[3])
    elif len(sys.argv) == 3:
        image_nb = sys.argv[1]
        image_path = f'./data/object_enhancement/{image_nb}_in.jpg'
        mask_path = f'./data/object_enhancement/masks/{image_nb}_mask.jpg'
        delta_s = float(sys.argv[2])
    else:
        raise Exception("Usage: python manipulating_saliency_main.py <image_path> <mask_path> <delta_s>")
    
    # Check that the path corresponds to existing files
    if not os.path.isfile(image_path) or not os.path.isfile(mask_path):
        raise FileNotFoundError(f"Path {image_path} does not exist.")

    # Read the image
    input_image = cv2.imread(image_path)
    print("\n - Image size:", input_image.shape)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Read Mask    
    mask_image = cv2.imread(mask_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
    
    mask_image = cv2.resize(mask_image, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_NEAREST)
    assert input_image.shape[:2] == mask_image.shape[:2], "Image and mask must match in size"

    # Compute the pyramids
    pyramids, laplacian = get_pyramids(input_image, 2)
    mask_pyramids, mask_laplacian = get_pyramids(mask_image, 2)
    
    # First, we run the full algorithm on the smallest image of the pyramid
    img = pyramids[-1]
    mask_image = mask_pyramids[-1]

    # Rebinearize the mask
    _, mask_image = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
    mask_image[mask_image > 0] = 1

    # Starting with the coarsest image
    utils.header_print("\nRunning the algorithm on the coarsest image...")
    
    coarse_image, tau_positive, tau_negative = manipulate_saliency(img, mask_image, delta_s, max_iteration=10)
    utils.display_image(coarse_image, pyramids[-1])
    pyramids[-1] = coarse_image
    
    # Now, we use those tau + and tau - to run the algorithm on the images at finer scales without having to compute them again
    n = len(pyramids) - 1
    utils.header_print("\nRunning the algorithm on the finer images...")
    
    for i in range(n):
        # We take the image at one scale finer and get it back to the original size
        img = pyramids[n - i]
        lap = laplacian[n - 1 - i]
        # print the sizes
        reconstruced = reconstruct(img, lap)
        mask_image = mask_pyramids[n - i - 1]

        # Rebinearize the mask
        _, mask_image = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
        mask_image[mask_image > 0] = 1
                
        # We call the image update function
        img = image_update_only(reconstruced, mask_image, 2, tau_positive, tau_negative)
        
        # We put the image back in the pyramid
        pyramids[n - 1 - i] = img
        # display the image at this scale
        plt.figure(figsize=(10,10))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
    
    
    utils.display_images(pyramids)
    final_image = img
    
    # Display the original image and the saliency map
    utils.display_images([input_image, final_image, mask_image], ["intput", "modified", "mask"])
    # save the orinal image
    # save_image(image, folder_path + "original_image.jpg")

    plt.show()
    