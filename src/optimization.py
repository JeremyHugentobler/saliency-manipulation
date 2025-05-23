import cv2
import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import diags
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from skimage.util import view_as_windows

from src.utils import display_images

# import matplotlib.pyplot as plt

STRIDE = 1
MAX_ITERATION = 10
PATCH_MATCH_MAX_ITER = 2*10 # each "iterartion" is (rdm search + propagate)
EPSILON = 1e5

def minimize_J_global_poisson(J, R, d_positive, d_negative, d_pos_mask, d_neg_mask, patch_size, lambda_factor=0.5):
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

    # Pad the image so that all pixel location of J is a valid patch center.
    if radius > 0:
        J_paded = np.pad(J, ((radius, radius), (radius, radius), (0,0)), mode='reflect')
    else:
        J_paded = J

    off_field = generate_random_offset_field(R, J_paded, patch_size, d_positive, d_negative, d_pos_mask, d_neg_mask)
    print("    - Offset field initialized")
    ### SEARCH STEP
    print("    - Search step")
    off_field = minimize_off_field_dist(off_field, J_paded, R, patch_size, d_pos_mask, d_neg_mask)

    ### VOTING STEP
    print("    - Vote step")
    # Blow up image into shape (w,h,p,p,3)
    J_patches = view_as_windows(J_paded, (patch_size, patch_size, 3))
    J_patches = J_patches.squeeze()
    
    idxs = np.indices((width,height))
    l_x, l_y = idxs + off_field[:,:,:2].transpose(2,0,1).astype(np.int32)

    # Retrieve patches from location
    offseted_patches = J_patches[l_x, l_y]

    # Compute the mean of each patch
    # J_patched = J[l_x, l_y]
    per_pixel_patch_mean = offseted_patches.mean(axis=(2,3))

    # repad the mean image
    per_pixel_patch_mean_padded = np.pad(per_pixel_patch_mean, ((radius, radius), (radius, radius), (0, 0)), mode='reflect')

    # Blow up image into shape (w,h,p,p,3) to get overlaping mean
    mean_patches = view_as_windows(per_pixel_patch_mean_padded, (patch_size, patch_size, 3))
    mean_patches = mean_patches.squeeze()

    # compute the patch mean of the per pixel mean
    J_patched = mean_patches.mean(axis=(2,3))
    
    ### SCREEN-POISSON
    print("  - Applying Poisson Screening...")

    J_out = screen_poisson(J, J_patched, lambda_factor=lambda_factor)

    # print("\033[A\033[K\033[A\033[K", end="")

    return J_out

def get_patch(J,x,y,patch_size):
    """ retrieve the patch with top left corner's coordinate = (x,y), check if valid too"""
    if patch_size > 1:
        patch = J[x:x+patch_size, y:y+patch_size]

        # sanity check
        w,h = patch.shape[:2]
        assert w == patch_size and h == patch_size
    else:
        patch = J[x,y]
    return patch

def compute_dist(J, off_field, patch_size):
    w, h = off_field.shape[:2]

    # Blow up image into shape (w,h,p,p,3)
    J_patches = view_as_windows(J, (patch_size, patch_size, 3))
    J_patches = J_patches.squeeze()

    # Retrieve location indices from offsets
    idxs = np.indices((w,h))
    l_x, l_y = idxs + off_field[:,:,:2].transpose(2,0,1).astype(np.int32)

    # Clean indicies
    outsiders = np.logical_or(
        np.logical_or(l_x < 0, l_x >= w),
        np.logical_or(l_y < 0, l_y >= h)
    )

    # Replace invalid location with self
    l_x[outsiders] = idxs[0, outsiders]
    l_y[outsiders] = idxs[1, outsiders]

    # Retrieve patches from location
    offseted_patches = J_patches[l_x, l_y]

    # Compute SSD
    dists = np.sum((J_patches.astype(np.float64) - offseted_patches.astype(np.float64)) ** 2, axis=(2,3,4))

    # Correct the outsiders to be invalidated later
    if len(outsiders) > 0:
        dists[outsiders] = float('inf')

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
    # offset_field[R] = f_in[R]
    # offset_field[np.logical_not(R)] = f_out[np.logical_not(R)]

    # Compute the patch_distance
    dists = compute_dist(J, offset_field, patch_size)

    # return the offset field with the corresponding distance
    return np.concatenate((offset_field.transpose(2,0,1), dists[None,:,:])).transpose(1,2,0)

def random_offset_field_jitter(offset_field, radius=5):
    '''
    Create an offset field that randomly jitters the one specified, based on the range it is given
    '''
    w,h = offset_field.shape[:2]

    # Create a jitter matrix to offset the offset field
    jitter = np.random.randint(-radius, radius, (w,h,2))

    # apply the jitter
    new_offset_f = offset_field.copy()
    new_offset_f[:,:,:2] += jitter

    return new_offset_f    

def validated_candidates(off_field, candidates, J, patch_size, R, d_pos_mask, d_neg_mask):

    old_off_f = off_field.copy()

    # Recompute new SSD from candidates
    dists = compute_dist(J, candidates, patch_size)
    candidates[:,:,2] = dists
    
    # Gather all the points where the candidates are better
    better_offset = candidates[:,:,2] < off_field[:,:,2]

    # Update the off_field correspondigly
    off_field[better_offset] = candidates[better_offset]
    
    # check if new location is in right database
    w,h = off_field.shape[:2]
    idxs = np.indices((w,h))
    l_x, l_y = idxs + off_field[:,:,:2].transpose(2,0,1).astype(np.int32)

    wrong_db = np.logical_or(
        np.logical_and(R, np.logical_not(d_pos_mask[l_x, l_y])),                        # in mask, outside db+
        np.logical_and(np.logical_not(R), np.logical_not(d_neg_mask[l_x, l_y]))         # not in mask, outside db-
    )

    off_field[wrong_db] = old_off_f[wrong_db]

    return off_field

def minimize_off_field_dist(off_field, J, R, patch_size, d_pos_mask, d_neg_mask):
    """
    minimize an offset field F st, 
        fa F(x,y) = (x_off, y_off, d): d = SSD(patch(x,y), patch(x+x_off, y+y_off)) is minimized
    Note that if (x,y) in R, (x+x_off, y+y_off) is also in R (similar with not R)
    """
    width, height = R.shape

    # Storing each iteration's reconstruction
    images = []

    idxs = np.indices((width, height))
    val = (idxs.transpose(1,2,0) + off_field[:,:,:2]).transpose(2,0,1).astype(np.float32)

    images.append(cv2.remap(J, val[1], val[0], cv2.INTER_LINEAR))

    p_mode = 0
    r_max = min(width, height) // 4
    # Iterate and alternate between search and propagate
    for i in tqdm(range(PATCH_MATCH_MAX_ITER)):          
        if i%2 == 0:
            ### propagate mode
            candidates = propagate(off_field, R, p_mode)
                
            off_field = validated_candidates(off_field, candidates, J, patch_size, R, d_pos_mask, d_neg_mask)

            p_mode = 1-p_mode
                
        else:
            ### Random search mode
            r = r_max
            while r > 1:
                # compare with randomly generated offset field
                candidates = random_offset_field_jitter(off_field, radius=r)

                off_field = validated_candidates(off_field, candidates, J, patch_size, R, d_pos_mask, d_neg_mask)
                
                r = r//2        # halves the search radius for next iteration

        l_x, l_y = (idxs + off_field[:,:,:2].transpose(2,0,1)).astype(np.float32)

        assert not np.any(np.logical_or(
            np.logical_or(l_x < 0, l_x >= width),
            np.logical_or(l_y < 0, l_y >= height)
        )), "Wrong offset got accepted as valid"
        images.append(cv2.remap(J, l_y, l_x, cv2.INTER_NEAREST))        
    
    # images = [Image.fromarray(img.astype(np.uint8)) for img in images]
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
    outside_points = np.logical_or(
        np.logical_or(locations[:,:,0] < 0, locations[:,:,0] >= w),
        np.logical_or(locations[:,:,1] < 0, locations[:,:,1] >= h),
    )

    # and put them back to their old previous value
    off_field[outside_points] = old_of[outside_points]

    return off_field

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


    ### Test with full image screening
    # def A(x):

    #     #lambda * f - grad^2(x)
    #     lap = cv2.Laplacian(x.reshape(n,m,3).astype(np.float32), cv2.CV_32F)
    #     return lambda_factor * x - lap.flatten()

    # res, _ = cg(LinearOperator((n*m*3, n*m*3), matvec=A), b.flatten(), x0=J.flatten().astype(np.float64))
    # res = res.reshape((n,m,3)

    res = np.clip(res, 0, 255)
    res = np.floor(res).astype(np.uint8)
 
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
    To reconstruct, use the reconstruct function.
    
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
    upscaled_image = cv2.pyrUp(downscaled_image, dstsize=laplacian_higher.shape[:2][::-1])

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
    off_field = minimize_off_field_dist(off_field, J_paded, R, patch_size, d_positive, d_negative, d_pos_mask, d_neg_mask)

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

def screen_poisson_brightness(J, J_modified, lambda_factor):
    """
    Poisson blending only on grayscale brightness (mean RGB channel), preserving color.
    """
    J = J.astype(np.float32)
    J_modified = J_modified.astype(np.float32)
    n, m, _ = J.shape

    J_gray = np.mean(J, axis=2)
    J_mod_gray = np.mean(J_modified, axis=2)

    lap = cv2.Laplacian(J_gray, cv2.CV_32F)
    b = lambda_factor * J_mod_gray - lap

    def A(x):
        x_reshaped = x.reshape((n, m)).astype(np.float32)
        lap = cv2.Laplacian(x_reshaped, cv2.CV_32F)
        return lambda_factor * x - lap.flatten()

    x_blended, _ = cg(LinearOperator((n*m, n*m), matvec=A), b.flatten(), x0=J_gray.flatten())
    blended_gray = x_blended.reshape((n, m))

    # Merge new gray channel into original RGB by matching mean
    blended = J.copy()
    gray_ratio = (blended_gray + 1e-5) / (np.mean(J, axis=2) + 1e-5)
    for c in range(3):
        blended[:, :, c] *= gray_ratio

    return np.clip(blended, 0, 255).astype(np.uint8)

def screen_poisson_luminance(J, J_modified, lambda_factor):
    """
    Poisson blending only on perceptual luminance (Y from YUV), preserving color.
    """
    J = J.astype(np.float32)
    J_modified = J_modified.astype(np.float32)
    n, m, _ = J.shape

    # Y = 0.299R + 0.587G + 0.114B
    J_lum = 0.299 * J[:, :, 0] + 0.587 * J[:, :, 1] + 0.114 * J[:, :, 2]
    J_mod_lum = 0.299 * J_modified[:, :, 0] + 0.587 * J_modified[:, :, 1] + 0.114 * J_modified[:, :, 2]

    lap = cv2.Laplacian(J_lum, cv2.CV_32F)
    b = lambda_factor * J_mod_lum - lap

    def A(x):
        x_reshaped = x.reshape((n, m)).astype(np.float32)
        lap = cv2.Laplacian(x_reshaped, cv2.CV_32F)
        return lambda_factor * x - lap.flatten()

    x_blended, _ = cg(LinearOperator((n*m, n*m), matvec=A), b.flatten(), x0=J_lum.flatten())
    blended_lum = x_blended.reshape((n, m))

    # Adjust RGB by scaling each channel to match new luminance
    lum_ratio = (blended_lum + 1e-5) / (J_lum + 1e-5)
    blended = J.copy()
    for c in range(3):
        blended[:, :, c] *= lum_ratio

    return np.clip(blended, 0, 255).astype(np.uint8)

def adaptive_tau_initialization(s_map, mask, boost_factor=0.2):
    """
    Initializes tau_positive and tau_negative to favor small salient regions.
    
    Args:
        s_map: (H, W) saliency map, float32 or float64
        mask: (H, W) binary mask, uint8 or bool
        boost_factor: amount to amplify the contrast when foreground is small (e.g., 0.2)

    Returns:
        tau_positive, tau_negative
    """
    # Normalize saliency map if needed
    s_map = (s_map - s_map.min()) / (s_map.max() - s_map.min() + 1e-8)

    saliency_fg = s_map[mask > 0]
    saliency_bg = s_map[mask == 0]

    mean_fg = np.mean(saliency_fg)
    mean_bg = np.mean(saliency_bg)

    # Mask coverage ratio
    mask_ratio = np.sum(mask > 0) / mask.size

    # Contrast-driven adjustment
    contrast = mean_fg - mean_bg
    bias = (1 - mask_ratio) * boost_factor * np.sign(contrast)

    # Base taus at 70th/30th percentiles
    base_tau_pos = np.quantile(saliency_fg, 0.7)
    base_tau_neg = np.quantile(saliency_bg, 0.3)

    # Apply bias
    tau_positive = np.clip(base_tau_pos + bias, 0, 1)
    tau_negative = np.clip(base_tau_neg - bias, 0, 1)

    return tau_positive, tau_negative
