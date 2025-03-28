import cv2
import numpy as np
from scipy.sparse.linalg import cg

STRIDE = 7

def minimize_J_global_poisson(J, R, d_positive, d_negative, patch_size=7):
    """
    Core function that tries to minimize the following energy function:
        E(J,D+,D-) = E+ + E- + E_delta
    where:  E+ = sum{p in R}( min{q in D+} (D(p,q)) )
            E- = sum{p not in R}( min{q in D+} ((D(p,q)) )
            E_delta = ||grad_J - grad_I||
    Apply poisson screening on the global image
    """
    height, width, _ = J.shape
    J_patched = np.zeros_like(J)
    radius = patch_size // 2

    ### PATCH-MATCH

    # Pad the image so that the patch don't overlap.
    J_paded = np.pad(J, radius, mode="reflect")

    # Loop through all upper-left corners of each patch
    for y in range(0, height - patch_size, STRIDE):
        for x in range(0, width - patch_size, STRIDE):
            patch = J_patched[y:y+patch_size, x:x+patch_size]
            mask_patch = R[y:y+patch_size, x:x+patch_size]

            if R[y + radius, x + radius] > 0 and len(d_positive) > 0:
                # Inside the mask (more salient)
                best_match = find_best_match(patch, d_positive, J_paded)
                best_match *= mask_patch > 0                # keep only best match inside the mask
                best_match += patch * (mask_patch == 0)     # add back the original image outside the mask

            elif len(d_negative) > 0:
                # Outside the mask (less salient)
                best_match = find_best_match(patch, d_negative, J_paded)
                best_match *= mask_patch == 0                # keep only best match outside the mask
                best_match += patch * (mask_patch > 0)     # add back the original image inside the mask
            else:
                best_match = patch  # If DB is empty keep orignal patch
            

            J_patched[y:y+patch_size, x:x+patch_size] = best_match  # Replace patch

    ### SCREEN-POISSON

    # TODO: see if it's ok to apply poisson screening on unpadded image
    J_patched_paded = np.pad(J_patched, radius, mode="reflect")
    J_patched = screed_poisson(J_paded, J_patched_paded, lambda_factor = 5.0)

    # un-pad the image
    return J_patched_paded[radius:-radius, radius:-radius]

def minimize_J_global_poisson(J, R, d_positive, d_negative, patch_size=7):
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
    return np.sum((patch1.astype("float") - patch2.astype("float")) ** 2)

def find_best_match(patch, database, J):
    """
    Finds the most similar patch in the database using SSD (brute force search).
    """
    if len(database) == 0:
        return patch  # Return original if no match available
    
    min_ssd = float("inf")
    best_patch = patch
    patch_size = patch.shape[0]
    r = patch_size // 2
    
    #TODO: The paper talks about a better way of doing that.
    for x, y in database:
        x -= r
        y -= r
        candidate_patch = J[x:x+patch_size, y:y+patch_size]
        ssd = compute_SSD(patch, candidate_patch)
        if ssd < min_ssd:
            min_ssd = ssd
            best_patch = candidate_patch  # Store best match

    return best_patch


def screed_poisson(J, J_modified, lambda_factor = 5.0):
    """
    Screened Poisson optimization to smoothly blend the patched image with the original image.

    Args:
        J: The original image
        J_modified: The image with patched regions
        lambda_factor: The lambda factor for the optimization
    Returns:    
        The blended image
    """
    laplacian = cv2.Laplacian(J_modified, cv2.CV_64F)
    b = laplacian - lambda_factor * (J_modified - J)
    blended, _ = cg(lambda x: cv2.Laplacian(x, cv2.CV_64F) - lambda_factor * x, b.flatten())
    return blended.reshape(J.shape)

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