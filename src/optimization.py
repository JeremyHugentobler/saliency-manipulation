import cv2
import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import diags
import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt

STRIDE = 1
MAX_ITERATION = 10
PATCH_MATCH_MAX_ITER = 2*5 # each "iterartion" is (rdm search + propagate)

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

    # Pad the image so that all pixel of J is a possible patch center. (dont look, just know it works)
    J_paded = np.stack([np.pad(J[:,:,i], radius, mode="reflect") for i in range(3)]).transpose(1,2,0)
    
    # Create the mapping from a pixel location to (the current mean of the overlapping patches, number of elements. 
    searched_patches_map = np.zeros((J_paded.shape[0], J_paded.shape[1], 4)) # 4 = (curr_r_mean, curr_g_mean, curr_b_mean, # seen pixels)

    off_field = generate_random_offset_field(R, J_paded, patch_size)

    ### SEARCH STEP
    for i in range(2):
        print("   - Propagating "+("inside" if i%2==0 else "outside")+" of R")
        region = R if i==0 else -(R-255)
        database = d_positive if i==0 else d_negative
        off_field = minimize_off_field_dist(off_field, J_paded, region, patch_size, database)

    # Loop through all upper-left corners of each patch
    for x in range(0, width, STRIDE):
        for y in range(0, height, STRIDE):
            # patch = J_paded[x:x+patch_size, y:y+patch_size]
            # # mask_patch = R[x:x+patch_size, y:y+patch_size]

            # if R[x, y] > 0 and len(d_positive) > 0:
            #     # Inside the mask (more salient)
            #     best_match = random_patch_search_SSD(patch, d_positive, J_paded)

            # elif len(d_negative) > 0:
            #     # Outside the mask (less salient)
            #     best_match = random_patch_search_SSD(patch, d_negative, J_paded)
            # else:
            #     best_match = patch  # If DB is empty keep orignal patch
            bm_x, bm_y = (x,y) + off_field[x,y,:2].astype(np.int32)
            best_match = J_paded[bm_x:bm_x+patch_size, bm_y:bm_y+patch_size]
            
            ### VOTING STEP

            # Set of pixel that need to be updated by the mean value of the patchmatch res
            s = searched_patches_map[x:x+patch_size, y:y+patch_size]

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
    # J_patched_padded = screen_poisson(J_paded, J_patched_padded, lambda_factor=lambda_factor)

    print("\033[A\033[K\033[A\033[K", end="")
    # un-pad the image
    return np.floor(J_patched_padded[radius:-radius, radius:-radius]).astype(np.uint8)

def compute_SSD(patch1, patch2):
    """Computes Sum of Square Distance (SSD) between two patches of the same shape."""
    if patch1.shape != patch2.shape:
        return np.inf  # Incompatible patches — skip them
    return np.sum((patch1.astype(np.float64) - patch2.astype(np.float64)) ** 2)

def generate_random_offset_field(R, J, patch_size):
    """
    Gernerate a new random offset field for each pixel such that, pixels belonging to a region (in R or not in R) have 
    their random offset that is pointing towards an other pixel that is in the same region. 
    Also compute the distance between each pixel's patch and its generated offsetted patch
    """
    w, h = R.shape
    r_in = np.array(np.where(R>0)).T
    r_out = np.array(np.where(R==0)).T

    offset_field = np.zeros((w,h,2), dtype=np.int32)

    # Make 2 images, one f_in(x,y) st (x,y) in R, the other f_out(x,y) st (x,y) not in R; for all (x,y)
    random_1D = np.random.randint(0, len(r_in), (w,h))
    f_in = r_in[random_1D]

    random_1D = np.random.randint(0, len(r_out), (w,h))
    f_out = r_out[random_1D]

    # Merge the 2 images according to R
    offset_field[R>0] = f_in[R>0]
    offset_field[R==0] = f_out[R==0]
    
    # From selected point to offset
    idxs = np.indices((w,h)).transpose(1,2,0)
    offset_field -= idxs

    # Compute the patch_distance
    dists = np.zeros((w,h))

    for i in range(w):
        for j in range(h):
            patch = J[i:i+patch_size, j:j+patch_size]
            off_x, off_y = offset_field[i,j] + (i,j)
            off_patch = J[off_x:off_x+patch_size, off_y:off_y+patch_size]
            dists[i,j] = compute_SSD(patch, off_patch)

    return np.concatenate((offset_field.transpose(2,0,1), dists[None,:,:])).transpose(1,2,0)

def minimize_off_field_dist(off_field, J, R, patch_size, database):
    """
    minimize an offset field F st, 
        fa F(x,y) = (x_off, y_off, d): d = SSD(patch(x,y), patch(x+x_off, y+y_off)) is minimized
    Note that if (x,y) in R, (x+x_off, y+y_off) is also in R (similar with not R)
    """
    width, height = R.shape
    off_field_info = {
        "min": [],
        "max": [],
        "mean": [],
        "median": []
    }

    # Iterate and alternate between search and propagate
    for i in range(PATCH_MATCH_MAX_ITER):
        print(f"   - Iteration {i}")            
        if i%2 == 0:
            ### propagate mode
            p_mode = (i/2)%2
            off_field = propagate(off_field, R, p_mode)
            idxs = np.indices((width, height)).transpose(1,2,0)
            val = idxs + off_field[:,:,:2]
            if np.any(np.logical_or(val[:,:,0] >= width, val[:,:,1] >= height)):
                faulty = np.where((np.logical_or(val[:,:,0] >= width, val[:,:,1] >= height)))
                print(off_field[faulty])
                raise Exception("What the fuck ?")
                
        else:
            ### search mode
            for i in range(width):
                # if i%50==0:
                    # print(f"    - row {i} out of {width}")
                for j in range(height):
                    if not (R[i,j]>0):
                        continue
                    # max_r = min(width, height)
                    max_r = 8
                    off_field[i,j] = random_patch_search_radius(
                        (i,j), off_field[i,j,:2].astype(np.int32), off_field[i,j,2], 
                        patch_size, database, J, max_r
                    )
        off_field_info["max"].append(off_field[:,:,2].max())
        off_field_info["min"].append(off_field[:,:,2].min())
        off_field_info["mean"].append(off_field[:,:,2].mean())
        off_field_info["median"].append(np.median(off_field[:,:,2]))
    # plt.plot(off_field_info["max"], label="max")
    # plt.plot(off_field_info["min"], label="min")
    # plt.plot(off_field_info["mean"], label="mean")
    # plt.plot(off_field_info["median"], label="median")
    # plt.legend(loc="upper left")
    # plt.show()
    return off_field

def propagate(off_field, R, mode):
    """
    propaget an offset field in x and y direction, causal or anti causal given the mode selected.
    """
    w,h,_ = off_field.shape

    # initialization given if propagation is causal or anti-causal
    if mode == 0:
        delta = 1
        w_range = range(1,w)
        h_range = range(1,h)
    else:
        delta = -1
        w_range = range(w-2,0,-1)
        h_range = range(h-2,0,-1)
    
    for i in w_range:
        for j in h_range:
            if  not (R[i,j]>0):
                continue
            off_x, off_y, curr_dist = off_field[i,j]

            
            # x-wise neighbour test
            if  R[i-delta, j]>0:
                candidate_x, candidate_y, dist = off_field[i-delta, j]
                if dist < curr_dist and 0 <= i + candidate_x < w:
                    off_x = candidate_x
                    off_y = candidate_y
                    curr_dist = dist

            # y-wise neighbour test
            if  R[i, j-delta]>0:
                candidate_x, candidate_y, dist = off_field[i, j-delta]
                if dist < curr_dist and 0 < j + candidate_y < h:
                    off_x = candidate_x
                    off_y = candidate_y
                    curr_dist = dist
            
            # Store the best candidate
            off_field[i,j] = (off_x, off_y, curr_dist)
    return off_field


def random_patch_search_radius(p_coord, offset, curr_dist, patch_size, database, J, max_r):
    """
    Tries to randomly find a more similar patch to p than p + offset, arround p + offset.
    parms:
        p_coord: (x, y) coordinates of the patch's center to compare to
        offset: (x_off, y_off) offset that when added to p_coord give the location of curr most similar patch.
        curr_dist: SSD distance between p and p+offset
        patch_size: size of a patch
        database: The list of location of possible patch's center
        J: The image
        r: Max radius of search around p_coord + offset
    """
    # retreive relevent info
    x, y = p_coord + offset
    patch = J[x:x+patch_size, y:y+patch_size]

    r = max_r

    # iterate for each r/2 until r <= patch_size
    while r > 0:
        # generate new random offset to p_coord + offset within the radius r
        # TODO: mind the offset that lies outside the image
        new_off = offset + np.random.randint(-r, r, (2))

        x, y = p_coord + new_off
        
        # check if new coordinates are in DB
        if np.any(np.all(database == (x,y), axis=1)):
            candidate = J[x:x+patch_size, y:y+patch_size]
            candidate_dist = compute_SSD(patch, candidate)
            # if offseted patch better, update offset. o/w iterate
            if candidate_dist < curr_dist:
                curr_dist = candidate_dist
                offset = new_off
                break # more optimize to stop hear and propagate then to continue

        r = r//2
    return (offset[0], offset[1], curr_dist)


def random_patch_search_SSD(patch, database, J):
    """
    Randomly finds the most similar patch in the database using SSD.
    """
    if len(database) == 0:
        return patch  # Return original if no match available
    
    min_ssd = float("inf")
    best_patch = patch
    patch_size = patch.shape[0]

    for _ in range(MAX_ITERATION):
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
        blended, _ = cg(LinearOperator((n*m, n*m), matvec=A), b[:,:,c].flatten(), x0=J[:,:,c].flatten().astype(np.float64))
        res[:,:,c] = blended.reshape((n,m))
    return np.clip(res, 0, 255)

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
