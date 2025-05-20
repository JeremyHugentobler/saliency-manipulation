import cv2
import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import diags
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from numba import jit

from utils import display_images

# import matplotlib.pyplot as plt

STRIDE = 1
MAX_ITERATION = 10
PATCH_MATCH_MAX_ITER = 2*50 # each "iterartion" is (rdm search + propagate)
EPSILON = 1e5

def minimize_J_global_poisson(J, R, d_positive, d_negative, d_pos_mask, d_neg_mask, patch_size=7, lambda_factor=0.1):
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

    off_field = generate_random_offset_field(R, J_paded, patch_size, d_positive, d_negative, d_pos_mask, d_neg_mask)
    print("    - Offset field initialized")
    ### SEARCH STEP
    print("    - Search step")
    off_field = minimize_off_field_dist(off_field, J_paded, R, patch_size, d_pos_mask, d_neg_mask)

    ### VOTING STEP
    print("    - Vote step")
    # Loop through all upper-left corners of each patch
    for x in tqdm(range(0, width, STRIDE)):
        for y in range(0, height, STRIDE):
            bm_x, bm_y = (x,y) + off_field[x,y,:2].astype(np.int32)
            best_match = J_paded[bm_x:bm_x+patch_size, bm_y:bm_y+patch_size]
            

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

    J_patched_padded = screen_poisson(J_paded, J_patched_padded, lambda_factor=lambda_factor)

    # print("\033[A\033[K\033[A\033[K", end="")
    # un-pad the image
    return np.floor(J_patched_padded[radius:-radius, radius:-radius]).astype(np.uint8)

def compute_SSD(patch1, patch2) -> float:
    """Computes Sum of Square Distance (SSD) between two patches of the same shape."""
    if patch1.shape != patch2.shape:
        return np.inf  # Incompatible patches — skip them
    return np.sum((patch1 - patch2).astype(np.float64) ** 2)

@jit
def compute_dist(J, off_field, patch_size):
    w, h = off_field.shape[:2]

    dists = np.zeros((w,h))
    for i in range(w):
        for j in range(h):
            patch = J[i:i+patch_size, j:j+patch_size]
            off_x, off_y = off_field[i,j,:2]
            x = int(off_x + i)
            y = int(off_y + j)

            off_patch = J[x:x+patch_size, y:y+patch_size]
            dists[i,j] = compute_SSD(patch, off_patch)

    return dists

def generate_random_offset_field(R, J, patch_size, d_positive, d_negative, d_pos_mask, d_neg_mask):
    """
    Gernerate a new random offset field for each pixel such that, pixels belonging to a region (in R or not in R) have 
    their random offset that is pointing towards an other pixel that is in the same region. 
    Also compute the distance between each pixel's patch and its generated offsetted patch
    """
    w, h = R.shape

    offset_field = np.zeros((w,h,2), dtype=np.int32)

    # Find the converse of the intersection between DB and mask, 
    # as for pixels in the intersection, the offset can be zero (min dist = yourself)
    mask_DB_pos_inter = np.logical_and(R, np.logical_not(d_pos_mask))       # In mask but not in database +
    mask_DB_neg_inter = np.logical_not(np.logical_or(R, d_neg_mask))        # not in mask and not in database -

    # Make 2 images, one f_in(x,y) st (x,y) in d+, the other f_out(x,y) st (x,y) in D-; for all (x,y)
    random_db_idx = np.random.randint(0, len(d_positive), (w,h))
    f_in = d_positive[random_db_idx]
    

    random_db_idx = np.random.randint(0, len(d_negative), (w,h))
    f_out = d_negative[random_db_idx]


    # From selected point to offset
    idxs = np.indices((w,h)).transpose(1,2,0)
    f_in -= idxs
    f_out -= idxs

    # Merge the 2 images according to R and db's
    offset_field[mask_DB_pos_inter] = f_in[mask_DB_pos_inter]
    offset_field[mask_DB_neg_inter] = f_out[mask_DB_neg_inter]

    # Compute the patch_distance
    dists = compute_dist(J, offset_field, patch_size)

    # return the offset field with the corresponding distance
    return np.concatenate((offset_field.transpose(2,0,1), dists[None,:,:])).transpose(1,2,0)

def random_offset_field_jitter(offset_field, J, R, d_pos_mask, d_neg_mask, patch_size, radius=5):
    '''
    Create an offset field that randomly jitters the one specified, based on the range it is given
    '''
    w,h = offset_field.shape[:2]

    # Create a jitter matrix to offset the offset field
    jitter = np.random.randint(-radius, radius, (w,h,2))

    # apply the jitter
    new_offset_f = offset_field.copy()
    new_offset_f[:,:,:2] += jitter

    idxs = np.indices((w,h)).transpose(1,2,0)
    vals = idxs + new_offset_f[:,:,:2]
    
    # check if new location is in bound
    outside_points = np.where(np.logical_or(
        np.logical_or(vals[:,:,0] < 0, vals[:,:,0] >= w),
        np.logical_or(vals[:,:,1] < 0, vals[:,:,1] >= h),
    ))

    new_offset_f[outside_points] = offset_field[outside_points]


    # compute the new distances
    for i in range(w):
        for j in range(h):
            off_x, off_y, old_dist = new_offset_f[i,j]
            off_x = int(off_x + i)
            off_y = int(off_y + j)

            # check if new location is in right database
            db = d_pos_mask if R[i,j] > 0 else d_neg_mask
            if db[off_x, off_y] == 0:
                new_offset_f[i,j,2] = np.inf      # Set dist to max value to discard at next step
            else:
                patch = J[i:i+patch_size, j:j+patch_size]
                off_patch = J[off_x:off_x+patch_size, off_y:off_y+patch_size]
                new_offset_f[i,j,2] = compute_SSD(patch, off_patch)

    return new_offset_f    


def minimize_off_field_dist(off_field, J, R, patch_size, d_pos_mask, d_neg_mask):
    """
    minimize an offset field F st, 
        fa F(x,y) = (x_off, y_off, d): d = SSD(patch(x,y), patch(x+x_off, y+y_off)) is minimized
    Note that if (x,y) in R, (x+x_off, y+y_off) is also in R (similar with not R)
    """
    width, height = R.shape

    images = []

    idxs = np.indices((width, height))
    val = (idxs.transpose(1,2,0) + off_field[:,:,:2]).transpose(2,0,1).astype(np.float32)

    images.append(cv2.remap(J, val[1], val[0], cv2.INTER_LINEAR))

    p_mode = 0
    r = min(width, height)
    # Iterate and alternate between search and propagate
    for i in tqdm(range(PATCH_MATCH_MAX_ITER)):          
        if i%2 == 0:
            ### propagate mode
            candidates = propagate(off_field, R, p_mode)
            candidates[:,:,2] = compute_dist(J, candidates, patch_size)

            p_mode = 1-p_mode

            # Sanity check
            val = idxs.transpose(1,2,0) + off_field[:,:,:2]
            if np.any(np.logical_or(val[:,:,0] >= width, val[:,:,1] >= height)):
                faulty = np.where((np.logical_or(val[:,:,0] >= width, val[:,:,1] >= height)))
                print(off_field[faulty])
                raise Exception("What the fuck ?")
                
        else:
            # compare with randomly generated offset field
            # candidates = generate_random_offset_field(R, J, patch_size, d_positive, d_negative, d_pos_mask, d_neg_mask)
            candidates = random_offset_field_jitter(off_field, J, R, d_pos_mask, d_neg_mask, patch_size, radius=r)

            r = r//2        # halves the search radius for next iteration
            if r <= 1:
                r = min(width, height)

        # Gather all the points where the candidates are better
        better_offset = np.where(candidates[:,:,2] < off_field[:,:,2])

        # Update the off_field correspondigly
        off_field[better_offset] = candidates[better_offset]
        val = (idxs.transpose(1,2,0) + off_field[:,:,:2]).transpose(2,0,1).astype(np.float32)

        images.append(cv2.remap(J, val[1], val[0], cv2.INTER_LINEAR))
        
    # n_of_image = 9
    # display_images(images[-n_of_image:], [i for i in range(n_of_image)])
    
    images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_Lab2RGB).astype(np.uint8)) for img in images]

    images[0].save(
        './output/animation.gif',
        save_all=True,
        append_images=images[1:],
        duration=200,
        loop=0
    )

    return off_field

def propagate(off_field, R, mode):
    """
    propaget an offset field in x and y direction, causal or anti causal given the mode selected.
    """
    old_of = off_field.copy()
    if mode == 0:
        # Offset on x
        # gather mask of where the offsetted elements are better then the not ones (has to also be in the same region)
        x,y = np.where(np.logical_and(off_field[1:,:,2] < off_field[:-1,:,2], R[1:,:] == R[:-1,:]))
        off_field[x,y] = off_field[x+1,y]

        # Offset on y
        # gather mask of where the offsetted elements are better then the not ones (has to also be in the same region)
        x,y = np.where(np.logical_and(off_field[:,1:,2] < off_field[:,:-1,2], R[:,1:] == R[:,:-1]))
        off_field[x,y] = off_field[x,y+1]
    else:
        # Offset on x
        # gather mask of where the offsetted elements are better then the not ones (has to also be in the same region)
        x,y = np.where(np.logical_and(off_field[:-1,:,2] < off_field[1:,:,2], R[1:,:] == R[:-1,:]))
        off_field[x+1,y] = off_field[x,y]

        # Offset on y
        # gather mask of where the offsetted elements are better then the not ones (has to also be in the same region)
        x,y = np.where(np.logical_and(off_field[:,:-1,2] < off_field[:,1:,2], R[:,1:] == R[:,:-1]))
        off_field[x,y+1] = off_field[x,y]

    w, h = off_field.shape[:2]

    # find the offset that goes outside the image
    idxs = np.indices((w, h)).transpose(1,2,0)
    locations = idxs + off_field[:,:,:2]
    outside_points = np.where(np.logical_or(
        np.logical_or(locations[:,:,0] < 0, locations[:,:,0] >= w),
        np.logical_or(locations[:,:,1] < 0, locations[:,:,1] >= h),
    ))

    off_field[outside_points] = old_of[outside_points]

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
    laplacian = cv2.Laplacian(J.astype(np.float32), cv2.CV_32F)
    b = lambda_factor * J_modified.astype(np.float64) - laplacian
    res = np.zeros_like(b)

    def A(x):

        #lambda * f - grad^2(x)
        lap = cv2.Laplacian(x.reshape(n,m).astype(np.float32), cv2.CV_32F)
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


from collections import deque

def bfs_patchmatch_recursive_layered(x0, y0, J_padded, R, patch_size, off_field, image_layers, max_depth=999):
    visited = np.zeros(R.shape, dtype=bool)
    queue = deque()
    queue.append((x0, y0, 0))
    radius = patch_size // 2
    H, W = R.shape

    def get_flat_layer_index(x, y, patch_size):
        return (x % patch_size) + patch_size * (y % patch_size)

    def vote_patch(searched_patches_map, patch, x, y):
        r = patch.shape[0] // 2
        region = searched_patches_map[x - r:x + r + 1, y - r:y + r + 1]
        n = region[:, :, 3:4] + 1
        region[:, :, :3] = (region[:, :, :3] * region[:, :, 3:4] + patch) / n
        region[:, :, 3:4] = n

    while queue:
        x, y, depth = queue.popleft()

        if depth > max_depth or visited[x, y]:
            continue
        visited[x, y] = True

        bm_x, bm_y = (x, y) + off_field[x, y, :2].astype(np.int32)
        best_patch = J_padded[bm_x:bm_x+patch_size, bm_y:bm_y+patch_size]

        layer_id = get_flat_layer_index(x, y, patch_size)
        vote_patch(image_layers[layer_id], best_patch, x, y)

        for dx, dy in [(-patch_size, 0), (patch_size, 0), (0, -patch_size), (0, patch_size)]:
            nx, ny = x + dx, y + dy
            if radius <= nx < H - radius and radius <= ny < W - radius:
                queue.append((nx, ny, depth + 1))


def merge_layers(image_layers):
    H, W, _ = image_layers[0].shape
    final = np.zeros((H, W, 3), dtype=np.float32)
    weight = np.zeros((H, W, 1), dtype=np.float32)

    for layer in image_layers:
        final += layer[:, :, :3]
        weight += layer[:, :, 3:4]

    final = final / np.maximum(weight, 1e-5)
    return final, weight


def minimize_J_global_poisson_bfs(J, R, d_positive, d_negative, d_pos_mask, d_neg_mask, patch_size=7, lambda_factor=0.5):
    """
    Recursive BFS PatchMatch + layered voting + screened Poisson blending.
    Operates entirely in Lab space (no RGB conversion).
    Applies interior border fix, full padded edge overwrite, and vote fallback.

    Args:
        J: Input Lab image (HxWx3)
        R: Binary region mask
        d_positive: Coordinates in D+
        d_negative: Coordinates in D-
        d_pos_mask: Binary mask of D+
        d_neg_mask: Binary mask of D-
        patch_size: Patch size (odd)
        lambda_factor: Poisson blending strength

    Returns:
        result: Final Lab image (unpadded), dtype=uint8
        vote_weights: Confidence map of voted pixels
    """
    width, height, _ = J.shape
    radius = patch_size // 2

    print("  - Padding image...")
    J_paded = np.stack([np.pad(J[:, :, i], radius, mode="reflect") for i in range(3)]).transpose(1, 2, 0)

    print("  - Generating offset field...")
    off_field = generate_random_offset_field(R, J_paded, patch_size, d_positive, d_negative, d_pos_mask, d_neg_mask)

    print("  - Search step...")
    off_field = minimize_off_field_dist(off_field, J_paded, R, patch_size, d_pos_mask, d_neg_mask)

    print("  - BFS PatchMatch voting...")
    image_layers = [np.zeros((J_paded.shape[0], J_paded.shape[1], 4), dtype=np.float32) for _ in range(patch_size ** 2)]
    center_x, center_y = height // 2, width // 2
    bfs_patchmatch_recursive_layered(center_x, center_y, J_paded, R, patch_size, off_field, image_layers)

    print("  - Merging layers...")
    J_patched_padded, vote_weights = merge_layers(image_layers)

    print("  - Fixing interior borders in Lab space...")
    border = patch_size // 2 + 1
    lab_central = J_patched_padded[radius:-radius, radius:-radius]
    lab_original = J[radius:-radius, radius:-radius].copy()

    H_fix = min(lab_central.shape[0], lab_original.shape[0])
    W_fix = min(lab_central.shape[1], lab_original.shape[1])
    lab_central[:border, :W_fix] = lab_original[:border, :W_fix]
    lab_central[-border:, :W_fix] = lab_original[-border:, :W_fix]
    lab_central[:H_fix, :border] = lab_original[:H_fix, :border]
    lab_central[:H_fix, -border:] = lab_original[:H_fix, -border:]
    J_patched_padded[radius:-radius, radius:-radius] = lab_central

    print("  - Overwriting padded border ring with original Lab...")
    J_patched_padded[:radius, :] = J_paded[:radius, :]
    J_patched_padded[-radius:, :] = J_paded[-radius:, :]
    J_patched_padded[:, :radius] = J_paded[:, :radius]
    J_patched_padded[:, -radius:] = J_paded[:, -radius:]

    print("  - Replacing low-confidence votes with original Lab...")
    if vote_weights.ndim == 2:
        vote_mask = vote_weights > 1e-3
    else:
        vote_mask = vote_weights[:, :, 0] > 1e-3

    vote_mask_3c = np.expand_dims(vote_mask, axis=2)  # (H, W, 1)
    vote_mask_3c = np.tile(vote_mask_3c, (1, 1, 3))   # (H, W, 3)
    J_patched_padded = np.where(vote_mask_3c, J_patched_padded, J_paded)

    print("  - Screened Poisson blending...")
    J_patched_padded = screen_poisson(J_paded, J_patched_padded, lambda_factor=lambda_factor)

    print("  - Finalizing result...")
    result = np.floor(J_patched_padded[radius:-radius, radius:-radius]).astype(np.uint8)
    return result, vote_weights

