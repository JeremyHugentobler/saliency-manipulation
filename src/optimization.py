import cv2
import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import diags

# import matplotlib.pyplot as plt

STRIDE = 7
MAX_ITERATION = 40

def minimize_J_global_poisson(J, R, d_positive, d_negative, patch_size=7):
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

    ### PATCH-MATCH
    print("  - Applying PatchMatch...")

    # Pad the image so that the patch don't overlap. (dont look, just know it works)
    J_paded = np.stack([np.pad(J[:,:,i], radius, mode="reflect") for i in range(3)]).transpose(1,2,0)
    J_patched = np.zeros_like(J_paded)

    # Loop through all upper-left corners of each patch
    for x in range(0, width - radius-1, STRIDE):
        for y in range(0, height - radius-1, STRIDE):
            off_x = x + radius
            off_y = y + radius
            patch = J_paded[off_x:off_x+patch_size, off_y:off_y+patch_size]
            mask_patch = R[x:x+patch_size, y:y+patch_size]

            if R[x + radius, y + radius] > 0 and len(d_positive) > 0:
                # Inside the mask (more salient)
                best_match = find_best_match(patch, d_positive, J_paded)
                # best_match *= (mask_patch > 0)[:,:,None]               # keep only best match inside the mask
                # best_match += patch * (mask_patch == 0)[:,:,None]      # add back the original image outside the mask

            elif len(d_negative) > 0:
                # Outside the mask (less salient)
                best_match = find_best_match(patch, d_negative, J_paded)
                # best_match *= (mask_patch == 0)[:,:,None]           # keep only best match outside the mask
                # best_match += patch * (mask_patch > 0)[:,:,None]    # add back the original image inside the mask
            else:
                best_match = patch  # If DB is empty keep orignal patch
            
            
            J_patched[off_x:off_x+patch_size, off_y:off_y+patch_size] = best_match  # Replace patch

    crop = J_patched[radius:radius+width, radius:height+radius]
    J_patched_padded = np.stack([np.pad(crop[:,:,i], radius, mode="reflect") for i in range(3)]).transpose(1,2,0)
    
    ### SCREEN-POISSON
    print("  - Applying Poisson Screening...")

    # TODO: see if it's ok to apply poisson screening on unpadded image
    J_patched_padded = screed_poisson(J_paded, J_patched_padded, lambda_factor=5)

    print("\033[A\033[K\033[A\033[K", end="")
    # un-pad the image
    return np.floor(J_patched_padded[radius:radius+width, radius:height+radius]).astype(np.uint8)

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
    return np.sum((patch1 - patch2) ** 2)

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


def screed_poisson(J, J_modified, lambda_factor):
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

    laplacian = cv2.Laplacian(J, cv2.CV_64F)
    b = lambda_factor * J_modified - laplacian
    res = np.zeros_like(b)

    def A(x):

        #lambda * f - grad^2(x)
        lap = cv2.Laplacian(x.reshape(n,m), cv2.CV_64F)
        return lambda_factor * x - lap.flatten()

    for c in range(3):
        blended, _ = cg(LinearOperator((n*m, n*m), matvec=A), b[:,:,c].flatten(), x0=J[:,:,c].flatten())
        res[:,:,c] = blended.reshape((n,m))
    return res

    # b = J_modified - J

    # # Compute Laplacian Matrix
    # n, m, _ = b.shape

    # # Create the Laplacian operator for a 2D grid
    # main_diag = -4 * np.ones(n * m)
    # off_diag = np.ones(n * m - 1)
    # off_diag[m-1::m] = 0  # Remove horizontal neighbors that should not connect

    # # Create the sparse Laplacian matrix (for the 2D grid)
    # diagonals = [main_diag, off_diag, off_diag, off_diag, off_diag]
    # offsets = [0, -1, 1, -m, m]
    # L = diags(diagonals, offsets, shape=(n * m, n * m), format="csr")
    
    # for c in range(3):
    #     # Flatten the source term (b) to a 1D array
    #     b_flat = b[:,:,c].flatten()

    #     # Initial guess for phi (typically a zero array)
    #     phi_init = np.zeros_like(b_flat)

    #     # Solve the system A * phi = b using the Conjugate Gradient solver
    #     phi_flat, _ = cg(L, b_flat, x0=phi_init)

    #     J_modified[:,:,c] += lambda_factor * phi_flat.reshape(n, m)

    # return J_modified

    
    # blended, _ = cg(lambda x: cv2.Laplacian(x, cv2.CV_64F) - lambda_factor * x, b.flatten())

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
