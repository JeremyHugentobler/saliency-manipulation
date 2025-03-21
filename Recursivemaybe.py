import cv2
import numpy as np
from scipy.sparse.linalg import cg
from skimage.restoration import unwrap_phase

# --- Step 1: Compute Saliency Map ---
def compute_saliency_map(image):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliency_map = saliency.computeSaliency(image)
    return saliency_map if success else None

# --- Step 2: Extract High and Low Saliency Patches ---
def extract_saliency_patches(image, saliency_map, tau_positive, tau_negative, patch_size):
    """
    Extracts high-saliency (D+) and low-saliency (D-) patches based on the mean saliency of each patch.
    """
    height, width, _ = image.shape
    d_positive, d_negative = [], []
    
    for y in range(0, height - patch_size, patch_size):
        for x in range(0, width - patch_size, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]
            patch_saliency = np.mean(saliency_map[y:y+patch_size, x:x+patch_size])
            
            if patch_saliency >= tau_positive:
                d_positive.append(patch)
            elif patch_saliency <= tau_negative:
                d_negative.append(patch)
    
    return np.array(d_positive), np.array(d_negative)

# --- Step 3: PatchMatch Recursive Implementation ---
def patch_match_recursive(image, mask, saliency_map, tau_positive, tau_negative, delta_s, patch_size=7, iteration=0, max_iterations=20):
    """
    Recursive PatchMatch with Screened Poisson applied after each step, now operating on patches.
    """
    if iteration >= max_iterations:
        return image  # Stop recursion if max iterations reached
    
    # Extract saliency-aware patches
    d_positive, d_negative = extract_saliency_patches(image, saliency_map, tau_positive, tau_negative, patch_size)
    
    # Perform PatchMatch on patches
    height, width, channels = image.shape
    patched_image = image.copy()
    
    for y in range(0, height - patch_size, patch_size):
        for x in range(0, width - patch_size, patch_size):
            if mask[y, x] > 0 and len(d_positive) > 0:
                best_match = d_positive[np.random.randint(len(d_positive))]
            elif len(d_negative) > 0:
                best_match = d_negative[np.random.randint(len(d_negative))]
            else:
                continue
            
            patched_image[y:y+patch_size, x:x+patch_size] = best_match
    
    # Apply Screened Poisson Blending
    smoothed_image = solve_screened_poisson(patched_image, image)
    
    # Recompute saliency map
    new_saliency_map = compute_saliency_map(smoothed_image)
    if new_saliency_map is None:
        return smoothed_image  # Stop if saliency map computation fails
    
    # Update saliency contrast
    saliency_contrast = np.mean(new_saliency_map[mask > 0]) - np.mean(new_saliency_map[mask == 0])
    adjustment = np.abs(saliency_contrast - delta_s) * 0.1
    tau_positive += adjustment
    tau_negative -= adjustment
    
    if np.abs(saliency_contrast - delta_s) < 0.05:
        return smoothed_image  # Stop recursion if target contrast is met
    
    # Recursive call
    return patch_match_recursive(smoothed_image, mask, new_saliency_map, tau_positive, tau_negative, delta_s, patch_size, iteration + 1, max_iterations)

# --- Step 4: Screened Poisson Blending ---
def solve_screened_poisson(image, guide, lambda_factor=5.0):
    """
    Solves the screened Poisson equation to blend modified regions smoothly.
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    b = laplacian - lambda_factor * (image - guide)
    blended, _ = cg(lambda x: cv2.Laplacian(x, cv2.CV_64F) - lambda_factor * x, b.flatten())
    return blended.reshape(image.shape)

# --- Step 5: Saliency-Guided Image Manipulation ---
def manipulate_saliency(image, mask, delta_s=0.6, max_iterations=20, patch_size=7):
    """
    Main function to adjust saliency of an image based on user input.
    """
    saliency_map = compute_saliency_map(image)
    if saliency_map is None:
        raise ValueError("Saliency map computation failed.")
    
    tau_positive, tau_negative = 0.5, 0.5  # Initial thresholds
    
    # Call recursive PatchMatch function
    final_image = patch_match_recursive(image, mask, saliency_map, tau_positive, tau_negative, delta_s, patch_size, iteration=0, max_iterations=max_iterations)
    return final_image

# --- Example Usage ---
if __name__ == "__main__":
    input_image = cv2.imread("input.jpg")
    mask = cv2.imread("mask.jpg", cv2.IMREAD_GRAYSCALE)  # Load user-defined mask
    
    result = manipulate_saliency(input_image, mask)
    cv2.imwrite("output.jpg", result)
