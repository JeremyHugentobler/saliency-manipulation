import cv2
import numpy as np
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import diags
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from skimage.util import view_as_windows

from src.utils import display_images, VERBOSE
import logging

# import matplotlib.pyplot as plt

STRIDE = 1
MAX_ITERATION = 10
PATCH_MATCH_MAX_ITER = 2*5 # each "iterartion" is (rdm search + propagate)
EPSILON = 100

def minimize_J_global_poisson(J, original_I, R, d_positive, d_negative, d_pos_mask, d_neg_mask, patch_size, lambda_factor=3):
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
        I_paded = np.pad(original_I, ((radius, radius), (radius, radius), (0,0)), mode='reflect')
    else:
        J_paded = J
        I_paded = original_I

    # off_field = np.zeros((width, height, 3), dtype=np.int32)
    off_field = generate_random_offset_field(R, I_paded, J_paded, patch_size, d_positive, d_negative, d_pos_mask, d_neg_mask)
    print("    - Offset field initialized")
    ### SEARCH STEP
    print("    - Search step")
    off_field = minimize_off_field_dist(off_field, I_paded, J_paded, R, patch_size, d_pos_mask, d_neg_mask)

    ### VOTING STEP
    print("    - Vote step")

    # Blow up image into shape (w,h,p,p,3)
    I_patches = view_as_windows(I_paded, (patch_size, patch_size, 3))
    I_patches = I_patches.squeeze()
    
    idxs = np.indices((width,height))
    l_x, l_y = idxs + off_field[:,:,:2].transpose(2,0,1).astype(np.int32)

    # Retrieve patches from location
    # J_patched = J[l_x, l_y]
    offseted_patches = I_patches[l_x, l_y]

    # Compute the mean of each patch
    J_patched = offseted_patches.mean(axis=(2,3)).astype(np.uint8)
    
    ### SCREEN-POISSON
    print("  - Applying Poisson Screening...")
    
    poisson = J_patched.copy()
    for _ in range(10):
        poisson = screen_poisson(original_I, poisson, lambda_factor=lambda_factor)

    # mean correction
    mean_diff = poisson.mean(axis=(0, 1)) - original_I.mean(axis=(0, 1))
    J_out =  (poisson - mean_diff).astype(np.uint8)


    if VERBOSE:
        diff = np.abs(poisson - J)
        display_images([
            cv2.cvtColor(J_patched.astype(np.uint8), cv2.COLOR_Lab2RGB), 
            cv2.cvtColor(J_out, cv2.COLOR_Lab2RGB), 
            cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_Lab2RGB),])

    return J_out

def vote_image(J, patch_size):
    # repad the mean image
    r = patch_size // 2
    per_pixel_patch_mean_padded = np.pad(J, ((r, r), (r, r), (0, 0)), mode='reflect')

    # Blow up image into shape (w,h,p,p,3) to get overlaping mean
    mean_patches = view_as_windows(per_pixel_patch_mean_padded, (patch_size, patch_size, 3))
    mean_patches = mean_patches.squeeze()

    # compute the patch mean of the per pixel mean
    J_patched = mean_patches.mean(axis=(2,3)).astype(np.uint8)
    
    return J_patched

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

def compute_dist(I, J, off_field, patch_size):
    w, h = off_field.shape[:2]

    # Blow up image into shape (w,h,p,p,3)
    J_patches = view_as_windows(J, (patch_size, patch_size, 3))
    J_patches = J_patches.squeeze()

    # Blow up image into shape (w,h,p,p,3)
    I_patches = view_as_windows(I, (patch_size, patch_size, 3))
    I_patches = I_patches.squeeze()

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
    offseted_patches = I_patches[l_x, l_y]

    # Compute SSD
    dists = np.sum((J_patches.astype(np.float64) - offseted_patches.astype(np.float64)) ** 2, axis=(2,3,4))

    # Correct the outsiders to be invalidated later
    if len(outsiders) > 0:
        dists[outsiders] = float('inf')

    return dists

def generate_random_offset_field(R, I, J, patch_size, d_positive, d_negative, d_pos_mask, d_neg_mask):
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
    dists = compute_dist(I, J, offset_field, patch_size)

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

def validated_candidates(off_field, candidates, I, J, patch_size, R, d_pos_mask, d_neg_mask):

    old_off_f = off_field.copy()

    # Recompute new SSD from candidates
    dists = compute_dist(I, J, candidates, patch_size)
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

def minimize_off_field_dist(off_field, I, J, R, patch_size, d_pos_mask, d_neg_mask):
    """
    minimize an offset field F st, 
        fa F(x,y) = (x_off, y_off, d): d = SSD(patch(x,y), patch(x+x_off, y+y_off)) is minimized
    Note that if (x,y) in R, (x+x_off, y+y_off) is also in R (similar with not R)
    """
    width, height = R.shape

    # Storing each iteration's reconstruction
    images = []

    idxs = np.indices((width, height))
    val = (idxs.transpose(1,2,0) + off_field[:,:,:2]).transpose(2,0,1).astype(np.float32) + patch_size // 2

    images.append(cv2.remap(J, val[1], val[0], cv2.INTER_LINEAR))

    p_mode = 0
    r_max = min(width, height) // 4
    # Iterate and alternate between search and propagate
    for i in tqdm(range(PATCH_MATCH_MAX_ITER)):
        if i%2 == 0:
            ### propagate mode
            candidates = propagate(off_field, R, p_mode)                
            off_field = validated_candidates(off_field, candidates, I, J, patch_size, R, d_pos_mask, d_neg_mask)

            p_mode = 1-p_mode
                
        else:
            ### Random search mode
            r = r_max
            while r > 1:
                # compare with randomly generated offset field
                candidates = random_offset_field_jitter(off_field, radius=r)
                off_field = validated_candidates(off_field, candidates, I, J, patch_size, R, d_pos_mask, d_neg_mask)
                
                r = r//2        # halves the search radius for next iteration

        l_x, l_y = (idxs + off_field[:,:,:2].transpose(2,0,1)).astype(np.float32)

        assert not np.any(np.logical_or(
            np.logical_or(l_x < 0, l_x >= width),
            np.logical_or(l_y < 0, l_y >= height)
        )), "Wrong offset got accepted as valid"

        l_x += patch_size // 2
        l_y += patch_size // 2
        images.append(cv2.remap(J, l_y, l_x, cv2.INTER_NEAREST))
    
    # images = [Image.fromarray(img.astype(np.uint8)) for img in images]
    images = [Image.fromarray(cv2.cvtColor(img, cv2.COLOR_Lab2RGB).astype(np.uint8)) for img in images]

    images[0].save(
        './output/PM_animation.gif',
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
    new_off_field = off_field.copy()
    
    if mode == 0:
        # Offset on x
        # gather mask of where the offsetted elements are better then the not ones (has to also be in the same region)
        x,y = np.where(np.logical_and(off_field[1:,:,2] < off_field[:-1,:,2], R[1:,:] == R[:-1,:]))
        new_off_field[x,y] = off_field[x+1,y]

        # Offset on y
        # gather mask of where the offsetted elements are better then the not ones (has to also be in the same region)
        x,y = np.where(np.logical_and(off_field[:,1:,2] < off_field[:,:-1,2], R[:,1:] == R[:,:-1]))
        new_off_field[x,y] = off_field[x,y+1]
    else:
        # Offset on x
        # gather mask of where the offsetted elements are better then the not ones (has to also be in the same region)
        x,y = np.where(np.logical_and(off_field[:-1,:,2] < off_field[1:,:,2], R[1:,:] == R[:-1,:]))
        new_off_field[x+1,y] = off_field[x,y]

        # Offset on y
        # gather mask of where the offsetted elements are better then the not ones (has to also be in the same region)
        x,y = np.where(np.logical_and(off_field[:,:-1,2] < off_field[:,1:,2], R[:,1:] == R[:,:-1]))
        new_off_field[x,y+1] = off_field[x,y]

    w, h = off_field.shape[:2]

    # find the offset that goes outside the image
    idxs = np.indices((w, h)).transpose(1,2,0)
    locations = idxs + off_field[:,:,:2]
    outside_points = np.logical_or(
        np.logical_or(locations[:,:,0] < 0, locations[:,:,0] >= w),
        np.logical_or(locations[:,:,1] < 0, locations[:,:,1] >= h),
    )

    # and put them back to their old previous value
    new_off_field[outside_points] = off_field[outside_points]

    return new_off_field

def screen_poisson(gradient_source, J_modified, lambda_factor):
    """
    Screened Poisson optimization to smoothly blend the patched image with the original image.

    Args:
        J: The original image
        J_modified: The image with patched regions
        lambda_factor: The lambda factor for the optimization
    Returns:    
        The blended image
    """
    n, m, _ = gradient_source.shape
    laplacian = cv2.Laplacian(gradient_source.astype(np.float32), cv2.CV_32F)
    b = lambda_factor * J_modified.astype(np.float32) - laplacian
    res = np.zeros_like(b)

    def A(x):

        #lambda * f - grad^2(x)
        lap = cv2.Laplacian(x.reshape(n,m).astype(np.float32), cv2.CV_32F)
        return lambda_factor * x - lap.flatten()

    for c in range(3):
        blended, _ = cg(LinearOperator((n*m, n*m), matvec=A), b[:,:,c].flatten())
        res[:,:,c] = blended.reshape((n,m))

    res = np.floor(res).astype(np.uint8)
 
    return res

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
