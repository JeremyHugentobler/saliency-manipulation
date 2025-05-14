
import numpy as np
import matplotlib.pyplot as plt
import cv2, skimage, os, sys

# Local imports
from src import saliancy_map as sm
from src import utils
from src import optimization as opt
from src import database as db


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


def patch_match_database(image, mask, d_positive, d_negative, patch_size=7):
    """
    PatchMatch algorithm that replaces patches using the best match from a given database (D+ or D-).
    """
    height, width, _ = image.shape
    patched_image = image.copy()
    
    for y in range(0, height - patch_size, patch_size):
        for x in range(0, width - patch_size, patch_size):
            patch = patched_image[y:y+patch_size, x:x+patch_size]

            if mask[y, x] > 0 and len(d_positive) > 0:
                best_match = find_best_match(patch, d_positive)
            elif len(d_negative) > 0:
                best_match = find_best_match(patch, d_negative)
            else:
                continue  # Skip if no matching database is available
            
            patched_image[y:y+patch_size, x:x+patch_size] = best_match  # Replace patch

    return patched_image


def compute_mse(patch1, patch2):
    """Computes Mean Squared Error (MSE) between two patches."""
    return np.mean((patch1.astype("float") - patch2.astype("float")) ** 2)

def find_best_match_mse(patch, database):
    """
    Finds the most similar patch in the database using MSE (brute force search).
    """
    if len(database) == 0:
        return patch  # Return original if no match available
    
    min_mse = float("inf")
    best_patch = patch
    
    for candidate_patch in database:
        mse = compute_mse(patch, candidate_patch)
        if mse < min_mse:
            min_mse = mse
            best_patch = candidate_patch  # Store best match

    return best_patch




def compute_ssim(patch1, patch2):
    """Computes Structural Similarity Index (SSIM) between two patches."""
    return ssim(patch1, patch2, multichannel=True, data_range=patch2.max() - patch2.min())

def find_best_match_ssim(patch, database):
    """
    Finds the most similar patch in the database using SSIM.
    """
    if len(database) == 0:
        return patch  # Return original if no match available
    
    max_ssim = -1
    best_patch = patch
    
    for candidate_patch in database:
        score = compute_ssim(patch, candidate_patch)
        if score > max_ssim:
            max_ssim = score
            best_patch = candidate_patch  # Store best match

    return best_patch



def extract_saliency_patches(image, saliency_map, tau_positive, tau_negative, patch_size=7):
    """
    Extracts high-saliency (D+) and low-saliency (D-) patches based on the mean saliency of each patch.
    
    Parameters:
        - image: The input image (H, W, C).
        - saliency_map: The computed saliency map (H, W).
        - tau_positive: Threshold for high saliency (D+).
        - tau_negative: Threshold for low saliency (D-).
        - patch_size: Size of the patches.

    Returns:
        - d_positive: Array of high-saliency patches.
        - d_negative: Array of low-saliency patches.
    """
    height, width, _ = image.shape
    d_positive, d_negative = [], []

    for y in range(0, height - patch_size, patch_size):
        for x in range(0, width - patch_size, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]  # Extract patch
            patch_saliency = np.mean(saliency_map[y:y+patch_size, x:x+patch_size])  # Compute mean saliency
            
            if patch_saliency >= tau_positive:
                d_positive.append(patch)
            elif patch_saliency <= tau_negative:
                d_negative.append(patch)
    
    return np.array(d_positive), np.array(d_negative)



def update_thresholds(saliency_map, mask, tau_positive, tau_negative, delta_s, learning_rate=0.1):
    """
    Adjusts tau_positive and tau_negative to gradually reach the desired saliency contrast.

    Parameters:
        - saliency_map: The computed saliency map.
        - mask: Binary mask defining the target region.
        - tau_positive: Current threshold for high saliency.
        - tau_negative: Current threshold for low saliency.
        - delta_s: Target saliency contrast.
        - learning_rate: Adjustment factor.

    Returns:
        - Updated tau_positive and tau_negative.
    """
    # Compute current saliency contrast
    saliency_foreground = np.mean(saliency_map[mask > 0])  # Inside target region
    saliency_background = np.mean(saliency_map[mask == 0])  # Outside target region
    current_contrast = saliency_foreground - saliency_background

    # Adjust thresholds based on difference from desired contrast
    adjustment = np.abs(current_contrast - delta_s) * learning_rate
    tau_positive += adjustment
    tau_negative -= adjustment

    return tau_positive, tau_negative


def compute_saliency_map(image):
    saliency_map=sm.tempsal_saliency(image)

    return saliency_map



def manipulate_saliency(image, mask, delta_s=0.6, max_iterations=20, patch_size=7, blend_size=11):
    """
    Main function to manipulate saliency by modifying patches iteratively using PatchMatch.

    Parameters:
        - image: Input image (H, W, C).
        - mask: Binary mask (H, W), where 255 indicates the region of interest.
        - delta_s: Target saliency contrast.
        - max_iterations: Maximum iterations for iterative refinement.
        - patch_size: Size of PatchMatch patches.
        - blend_size: Size of Poisson blending region.

    Returns:
        - Final manipulated image with updated saliency.
    """
    # Compute saliency map
    saliency_map = compute_saliency_map(image)  #TO BE DEFINED#####################################
    if saliency_map is None:
        raise ValueError("Saliency map computation failed.")

    # Initialize thresholds
    tau_positive, tau_negative = 0.5, 0.5  # Initial values #####STILLNOTSURE########

    manipulated_image = image.copy()
    
    for iteration in range(max_iterations):
        # Extract high and low saliency patches
        d_positive, d_negative = extract_saliency_patches(image, saliency_map, tau_positive, tau_negative, patch_size)
        
        height, width, _ = manipulated_image.shape

        for y in range(0, height - patch_size, patch_size):
            for x in range(0, width - patch_size, patch_size):
                patch = manipulated_image[y:y+patch_size, x:x+patch_size]

                if mask[y, x] > 0 and len(d_positive) > 0:
                    #best_match = find_best_match_ssim(patch, d_positive)  
                    best_match = find_best_match_mse(patch, d_positive)  
                elif len(d_negative) > 0:
                    best_match = find_best_match_mse(patch, d_negative)  
                    #best_match = find_best_match_ssim(patch, d_negative)  
                else:
                    continue  # Skip if no database is available
                
                # Replace patch with best match
                manipulated_image[y:y+patch_size, x:x+patch_size] = best_match

                # Apply local Poisson blending with chosen blend size
                manipulated_image = local_screened_poisson(manipulated_image, best_match, (y, x, patch_size), blend_size)

        # Recompute saliency map after patch replacement
        saliency_map = compute_saliency_map(manipulated_image)

        # Update thresholds based on saliency contrast
        tau_positive, tau_negative = update_thresholds(saliency_map, mask, tau_positive, tau_negative, delta_s)

        # Compute saliency contrast
        saliency_foreground = np.mean(saliency_map[mask > 0])
        saliency_background = np.mean(saliency_map[mask == 0])
        saliency_contrast = saliency_foreground - saliency_background

        # Stop if contrast is close enough
        if np.abs(saliency_contrast - delta_s) < 0.05:
            break

    return manipulated_image


def main(image_path, mask_path, output_path, delta_s=0.6, patch_size=7, blend_size=11, max_iterations=2):
    """
    Entry point to run the full saliency manipulation pipeline.
    """
    # Load image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise FileNotFoundError("Image or mask could not be loaded.")

    # Run saliency-guided manipulation
    result = manipulate_saliency(
        image=image,
        mask=mask,
        delta_s=delta_s,
        max_iterations=max_iterations,
        patch_size=patch_size,
        blend_size=blend_size
    )

    # Save result
    cv2.imwrite(output_path, result)
    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    image_path = "output/Apple.png"
    mask_path = "output/AppleMask.png"
    output_path = "output/apple_modified.png"
    main(image_path, mask_path, output_path)