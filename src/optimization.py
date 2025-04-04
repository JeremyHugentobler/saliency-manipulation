import cv2
import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import diags

# import matplotlib.pyplot as plt

STRIDE = 3
MAX_ITERATION = 10

def minimize_J_global_poisson(J, R, d_positive, d_negative, patch_size=7, lambda_factor=5):
    """
    Core function that tries to minimize the following energy function:
        E(J,D+,D-) = E+ + E- + E_delta
    where:  E+ = sum{p in R}( min{q in D+} (D(p,q)) )
            E- = sum{p not in R}( min{q in D+} ((D(p,q)) )
            E_delta = ||grad_J - grad_I||
    Apply poisson screening on the global image
    """
    width, height, _ = J.shape
    radius = patch_size // 2

    ### SEARCH-AND-VOTE
    print("  - Applying PatchMatch...")

    # Pad the image so that the patch don't overlap. (dont look, just know it works)
    J_paded = np.stack([np.pad(J[:,:,i], radius, mode="reflect") for i in range(3)]).transpose(1,2,0)
    
    # Create the mapping from a pixel location to (the current mean of the overlapping patches, number of elements. 
    searched_patches_map = np.zeros((J_paded.shape[0], J_paded.shape[1], 4)) # 4 = (curr_r_mean, curr_g_mean, curr_b_mean, # seen pixels)

    # Loop through all upper-left corners of each patch
    for x in range(0, width, STRIDE):
        for y in range(0, height, STRIDE):
            patch = J_paded[x:x+patch_size, y:y+patch_size]
            # mask_patch = R[x:x+patch_size, y:y+patch_size]

            if R[x, y] > 0 and len(d_positive) > 0:
                # Inside the mask (more salient)
                best_match = find_best_match(patch, d_positive, J_paded)

            elif len(d_negative) > 0:
                # Outside the mask (less salient)
                best_match = find_best_match(patch, d_negative, J_paded)
            else:
                best_match = patch  # If DB is empty keep orignal patch
                
            # VOTING STEP (but done on the fly)

            # Set of pixel that need to be updated by the mean value of the patchmatch res
            s = searched_patches_map[x:x+patch_size, y:y+patch_size]

            # TODO: filter out what is not in the given "corect" region
            
            # new_mean (per pixel) = temp_mean * N/(N+p*p) + p_mean / N+p*p
            n_plus_p2 = (s[:,:,3] + patch_size**2)
            s[:,:,:-1] *= (s[:,:,3] / n_plus_p2)[:,:,None]
            s[:,:,:-1] += np.ones_like(best_match) * best_match.sum(axis=(0,1)) / n_plus_p2[:,:,None]

            # Update the intermidiate total number
            s[:,:,-1] = n_plus_p2.copy()

    # Remove temporary accumulator
    J_patched_padded = searched_patches_map[:,:,:-1]
    
    ### SCREEN-POISSON
    print("  - Applying Poisson Screening...")

    # TODO: see if it's ok to apply poisson screening on unpadded image
    J_patched_padded = screen_poisson(J_paded, J_patched_padded, lambda_factor=lambda_factor)

    print("\033[A\033[K\033[A\033[K", end="")
    # un-pad the image
    return np.floor(J_patched_padded[radius:-radius, radius:-radius]).astype(np.uint8)

def minimize_J_local_poisson(J, R, d_positive, d_negative, patch_size=7):
    """
    Core function that tries to minimize the following energy function:
        E(J,D+,D-) = E+ + E- + E_delta
    where:  E+ = sum{p in R}( min{q in D+} (D(p,q)) )
            E- = sum{p not in R}( min{q in D+} ((D(p,q)) )
            E_delta = ||grad_J - grad_I||
    Apply poisson screening on the global image
    """

    #TODO
    pass

def compute_SSD(patch1, patch2):
    """Computes Sum of Square Distance (SSD) between two patches."""
    return np.sum((patch1.astype(np.float64) - patch2.astype(np.float64)) ** 2)

def find_best_match(patch, database, J):
    """
    Finds the most similar patch in the database using SSD (brute force search).
    """
    if len(database) == 0:
        return patch  # Return original if no match available
    
    min_ssd = float("inf")
    best_patch = patch
    patch_size = patch.shape[0]
        
    #TODO: The paper talks about a better way of doing that.
    # max_iter = np.floor(np.sqrt(len(database))).astype(np.uint32)
    max_iter = MAX_ITERATION
    for _ in range(max_iter):
        # Random search
        x,y = database[np.random.randint(len(database))]

        candidate_patch = J[x:x+patch_size, y:y+patch_size]
        ssd = compute_SSD(patch, candidate_patch)
        
        # Optimization to return faster if the patch exist in DB
        # if ssd < 1e-7:
        #     return candidate_patch
        
        if ssd < min_ssd:
            min_ssd = ssd
            best_patch = candidate_patch  # Store best match
    
    return best_patch


def screen_poisson(J, J_modified, lambda_factor):
    """
    Screened Poisson optimization to smoothly blend the patched image with the original image.

    Args:
        J: The original image
        J_modified: The image with patched regions
        lambda_factor: The lambda factor for the optimization
    Returns:    
        The blended image
    """
    n, m, _ = J.shape

    laplacian = cv2.Laplacian(J.astype(np.float64), cv2.CV_64F)
    b = lambda_factor * J_modified.astype(np.float64) - laplacian
    res = np.zeros_like(b)

    def A(x):

        #lambda * f - grad^2(x)
        lap = cv2.Laplacian(x.reshape(n,m), cv2.CV_64F)
        return lambda_factor * x - lap.flatten()

    for c in range(3):
        blended, _ = cg(LinearOperator((n*m, n*m), matvec=A), b[:,:,c].flatten(), x0=J[:,:,c].flatten())
        res[:,:,c] = blended.reshape((n,m))
    return res

def local_screened_poisson(image, modified_patch, patch_coords, blend_patch_size):
    """
    Applies local screened Poisson blending to a selected patch with adjustable blending size.

    Parameters:
        - image: Original image (H, W, C).
        - modified_patch: The modified patch to be blended (p, p, C).
        - patch_coords: Tuple (y, x, patch_size) indicating the patch location.
        - blend_patch_size: Size of the region used for Poisson blending (must be larger than patch_size).

    Returns:
        - Blended image with locally applied Poisson blending.
    """
    y, x, patch_size = patch_coords

    # Ensure blend_patch_size is at least as large as patch_size
    blend_patch_size = max(blend_patch_size, patch_size + 2)

    # Compute region bounds, ensuring we stay within image dimensions
    half_blend = blend_patch_size // 2
    y1, y2 = max(0, y - half_blend), min(image.shape[0], y + patch_size + half_blend)
    x1, x2 = max(0, x - half_blend), min(image.shape[1], x + patch_size + half_blend)

    # Extract the blending region from the original image
    blend_region = image[y1:y2, x1:x2]

    # Create binary mask (white for modified region, black for background)
    mask = np.zeros_like(blend_region[:, :, 0], dtype=np.uint8)
    patch_y1, patch_y2 = y - y1, y - y1 + patch_size
    patch_x1, patch_x2 = x - x1, x - x1 + patch_size
    mask[patch_y1:patch_y2, patch_x1:patch_x2] = 255  # White inside the patch

    # Resize modified patch to match blend_patch_size
    resized_patch = cv2.resize(modified_patch, (blend_region.shape[1], blend_region.shape[0]))

    # Apply Poisson blending in seamlessClone mode
    blended_patch = cv2.seamlessClone(resized_patch, blend_region, mask, 
                                      (blend_region.shape[1] // 2, blend_region.shape[0] // 2), cv2.NORMAL_CLONE)

    # Replace the blended region back into the image
    blended_image = image.copy()
    blended_image[y1:y2, x1:x2] = blended_patch

    return blended_image


def get_pyramids(image, levels):
    """
    Generate Gaussian and Laplacian pyramids for the given image.
    To reconstruct, use the reconstruct function. Warning
    
    Args:
        image (numpy.ndarray): Input image.
        levels (int): Number of levels in the pyramid. (divide by 2 each time)

    Returns:
        tuple: Gaussian and Laplacian pyramids.
    """
    
    gaussian_pyramid = [image]
    laplacian_pyramid = []

    # Build Gaussian pyramid
    for i in range(levels):
        image = cv2.pyrDown(image)
        gaussian_pyramid.append(image)
    
    # From them, we can derive the laplacians
    for i in range(levels):
        original = gaussian_pyramid[i]
        downscaled = gaussian_pyramid[i+1]
        upscaled = cv2.pyrUp(downscaled, dstsize=original.shape[:2][::-1])
        
        laplacian = cv2.subtract(original, upscaled)
        laplacian_pyramid.append(laplacian)
    
    return gaussian_pyramid, laplacian_pyramid

def reconstruct(downscaled_image, laplacian_higher):
    """
    Reconstruct the image from its Laplacian pyramid and the downscaled image.

    Args:
        downscaled_image (numpy.ndarray): The downscaled image from the Gaussian pyramid.
        laplacian_higher (numpy.ndarray): The Laplacian image from the higher level.

    Returns:
        numpy.ndarray: The reconstructed image.
    """
    
    # Upscale the downscaled image to match the size of the Laplacian image
    upscaled_image = cv2.pyrUp(downscaled_image)

    reconstructed_image = cv2.add(upscaled_image, laplacian_higher)

    return reconstructed_image