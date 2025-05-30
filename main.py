# USAGE EXEMPLE: python .\main.py .\data\debug\easy_apple_small.jpg .\data\debug\easy_apple_small_mask.jpg 0.1

# Libraries imports
import numpy as np
import matplotlib.pyplot as plt
import cv2, skimage, os, sys
from PIL import Image

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
compute_saliency_map = sm.custom_saliency
minimize_J = opt.minimize_J_global_poisson
compute_database = db.compute_location_database



def manipulate_saliency(input_image, R, delta_s, max_iteration=10, patch_size=7, learning_rate=0.1, convert=True):
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
    tau_positive = 0.3
    tau_negative = 0.7

    print("\ninitalizing variables")

    # Initialize the interational images buffers (Try to keep them as [0,255] images)
    J = np.array([np.zeros_like(input_image) for _ in range(2)])
    J[0] = input_image
    
    # Initialize the saliency maps S_J
    print(" - computing saliency map...")
    S_J = compute_saliency_map(J[0])
    print(" - Done.")
    
    S_I = S_J.copy()  # Initial saliency map
    s_maps = [S_J.copy()]
    saliency_contrast = []
    images = [input_image.copy()]

    input_image_Lab = cv2.cvtColor(input_image, cv2.COLOR_RGB2Lab)
    
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
        D_positive, D_negative, D_pos_mask, D_neg_mask = compute_database(tau_positive, tau_negative, input_image, S_I)
        print(f" - Done, DB+ size: {D_positive.shape[0]}, DB- size: {D_negative.shape[0]}")
        
        if utils.VERBOSE:
            # Construct and display the database's images
            I_D_positive, I_D_negative = db.compute_image_database(input_image, D_positive, D_negative)

            utils.display_images([S_J, I_D_positive, I_D_negative])

        # update J to minimize the energy function
        print(" - Minimizing function...")
        
        # Convert image into Lab color space
        if convert:
            J[0] = cv2.cvtColor(J[0], cv2.COLOR_RGB2Lab)

        J[1] = minimize_J(J[0], input_image_Lab, R, D_positive, D_negative, D_pos_mask, D_neg_mask, patch_size)
        print(" - Done.")

        # Convert back into RGB
        if convert:
            J[0] = cv2.cvtColor(J[0], cv2.COLOR_Lab2RGB)
            J[1] = cv2.cvtColor(J[1], cv2.COLOR_Lab2RGB)

        # Compute new saliency map
        print(" - computing new saliency map...")
        S_J = compute_saliency_map(J[1])
        s_maps.append(S_J.copy())
        print(" - Done.")

        s_diff = S_J - s_maps[-2]
        if utils.VERBOSE:
            utils.display_images([J[0], J[1],S_J,s_diff])

        # Compute criterions
        criterion = compute_criterion(S_J, R, delta_s)
        saliency_contrast.append(phi(S_J, R))
        print(" - Iteration's saliency contrast:", phi(S_J, R), "Objective:", delta_s)

        # Check if convergence is reached by tau's
        tau_diff = learning_rate * criterion
        if tau_diff < EPSILON:
            print(" - Tau's convergence reached.")
            break
        
        # Check if convergence is reached by delta_s
        if criterion < EPSILON or phi(S_J, R) > delta_s:
            print(" - Criterion convergence reached.")
            break

        # Update tau +/-
        tau_positive, tau_negative = update_taus(tau_positive, tau_negative, criterion, learning_rate)

        images.append(J[1].copy())
        # switch the buffers (only affect the references so no copy is made)
        temp = J[0].copy()
        J[0] = J[1]
        J[1] = temp

    print("Done")
    return images, s_maps, tau_positive, tau_negative, saliency_contrast

def image_update_only(input_image, original_img, R, iterations, tau_positive, tau_negative, patch_size=7):
    """Called for image at finer scales, it's basically only the second part of 'manipulate_saliency' and refered as 'image update' in the paper.

    Args:
        input_image: The image (not the coarsest one) on which to apply the image update
        R: mask defining the region of interest
        iterations: number of iterations to run (5 to 20 mentionned in the paper)
        tau_positive : tau found in the manipulate_saliency function
        tau_negative : tau found in the manipulate_saliency function
        patch_size (int, optional): The patch size. Defaults to 7.
    """    
    
    # buffers
    images = []
    s_maps = []

    # Input in RGB, transform to Lab
    image = input_image.copy()
    original_img_lab = cv2.cvtColor(original_img, cv2.COLOR_RGB2Lab)

    S_I = compute_saliency_map(original_img)


    saliency_contrast = []
        
    for i in range(iterations):
        
        # Get the saliency map
        s_map = compute_saliency_map(image)
        s_maps.append(s_map.copy())
        saliency_contrast.append(phi(s_map, R))

        # Get the databases
        D_positive, D_negative, D_pos_mask, D_neg_mask = compute_database(tau_positive, tau_negative, original_img, S_I)
        
        # Convert in Lab
        image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        # Compute the image update
        
        image = minimize_J(image, original_img_lab, R, D_positive, D_negative, D_pos_mask, D_neg_mask, patch_size)

        # Convert back in RGB
        image = cv2.cvtColor(image, cv2.COLOR_Lab2RGB)

        images.append(image.copy())

    return images, s_maps, saliency_contrast

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

    tau_positive = min(tau_positive, 0.9)  # Ensure tau_positive is not too small
    tau_negative = max(tau_negative, 0.1)  # Ensure tau_negative is not too small

    return tau_positive, tau_negative
        

def main( image_path, mask_path, delta_s):
    """ Main function to run the saliency manipulation algorithm on an image with a mask.
    Args:
        image_path (str): Path to the input image.
        mask_path (str): Path to the mask image.
        delta_s (float): Desired saliency contrast between the region of interest and the rest of the image.
    """


    # Read the image
    input_image = cv2.imread(image_path)
    print("\n - Image size:", input_image.shape)
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    # Read Mask    
    mask_image = cv2.imread(mask_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)

    h, w = input_image.shape[:2]
    
    mask_image = cv2.resize(mask_image, (w, h), interpolation=cv2.INTER_NEAREST)
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

    # Buffers
    saliency_contrast = []
    s_maps = []
    images = []

    # Starting with the coarsest image
    utils.header_print("\nRunning the algorithm on the coarsest image...")
    
    coarse_images, coarse_s_maps, tau_positive, tau_negative, saliency = manipulate_saliency(img, mask_image, delta_s, max_iteration=10)
    coarse_image = coarse_images[-1].copy()

    # Save info
    images.extend(coarse_images)
    s_maps.extend(coarse_s_maps)
    saliency_contrast.extend(saliency)

    original_img = pyramids[-1].copy()
    pyramids[-1] = coarse_image.copy()
    
    # Now, we use those tau + and tau - to run the algorithm on the images at finer scales without having to compute them again
    n = len(pyramids) - 1
    utils.header_print("\nRunning the algorithm on the finer images...")
    
    for i in range(n):
        # more iteration at the beginning then at the end
        nb_of_iterations = ((n - i))**2

        # We take the image at one scale finer and get it back to the original size
        img = pyramids[n - i]
        lap = laplacian[n - 1 - i]
        # print the sizes
        reconstruced = reconstruct(img, lap)
        og_reconsructed = reconstruct(original_img, lap)
        if utils.VERBOSE:
            utils.display_images([og_reconsructed, reconstruced, s_maps[-1]], ["Original", "Reconstructed", "Saliency Map"])
        mask_image = mask_pyramids[n - i - 1]

        # Rebinearize the mask
        _, mask_image = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
        mask_image[mask_image > 0] = 1
                
        # We call the image update function
        iter_imgs, iter_s_maps, saliency = image_update_only(reconstruced, og_reconsructed, mask_image, nb_of_iterations, tau_positive, tau_negative)
        img = iter_imgs[-1].copy()

        # store infos
        saliency_contrast.extend(saliency)
        images.extend(iter_imgs)
        s_maps.extend(iter_s_maps)

        # We put the image back in the pyramid
        original_img = pyramids[n - 1 - i].copy()
        pyramids[n - 1 - i] = img
    
    # Save animations
    images = [cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR) for img in images]
    s_maps = [cv2.resize(s_map, (w, h), interpolation=cv2.INTER_LINEAR) for s_map in s_maps]

    images = [Image.fromarray(img) for img in images]
    s_maps = [Image.fromarray((s_map * 255).astype(np.uint8)) for s_map in s_maps]

    s_maps[0].save(
        './output/saliency_animation.gif',
        save_all=True,
        append_images=s_maps[1:],
        duration=200,
        loop=0
    )
    
    images[0].save(
        './output/animation.gif',
        save_all=True,
        append_images=images[1:],
        duration=200,
        loop=0
    )

    if utils.VERBOSE:
        utils.display_images(pyramids)
    final_image = img
    
    # Display the original image and the saliency map
    if utils.DISP_OUT:        
        input_s_map = compute_saliency_map(input_image)
        utils.display_images([input_image, final_image, mask_image, input_s_map, s_maps[-1]], ["intput", "modified", "mask", "input saliency map", "final saliency map"])

        plt.plot(saliency_contrast)
        plt.title("Saliency contrast over iterations")

        plt.show()
        
    return final_image

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
    
    main(image_path, mask_path, delta_s)
    
    