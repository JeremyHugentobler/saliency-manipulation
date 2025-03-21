import cv2
import numpy as np
from scipy.sparse.linalg import cg
from skimage.restoration import unwrap_phase


def compute_saliency_map(image):
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()  #for now ill use this but we need a model
    success, saliency_map = saliency.computeSaliency(image)
    return saliency_map if success else None




#Per pixel saliency
def extract_saliency_patches(image, saliency_map, tau_positive, tau_negative):
    """
    Extracts high-saliency (D+) and low-saliency (D-) patches based on thresholds.
    """
    d_positive = image[saliency_map >= tau_positive]
    d_negative = image[saliency_map <= tau_negative]
    return d_positive, d_negative


#Per patch saliency 
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



def patch_match(image, mask, d_positive, d_negative, patch_size=7, num_iterations=5):
    """
    Uses a randomized nearest-neighbor search to replace patches based on saliency constraints.
    """
    height, width, channels = image.shape
    patched_image = image.copy()
    
    for _ in range(num_iterations):
        for y in range(0, height - patch_size, patch_size):
            for x in range(0, width - patch_size, patch_size):
                patch = patched_image[y:y+patch_size, x:x+patch_size]
                
                if mask[y, x] > 0:
                    # Inside the mask - use high-saliency patches
                    best_match = d_positive[np.random.randint(len(d_positive))]
                else:
                    # Outside the mask - use low-saliency patches
                    best_match = d_negative[np.random.randint(len(d_negative))]
                
                patched_image[y:y+patch_size, x:x+patch_size] = best_match
    
    return patched_image


def solve_screened_poisson(image, guide, lambda_factor=5.0):
    """
    Solves the screened Poisson equation to blend modified regions smoothly.
    """
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    b = laplacian - lambda_factor * (image - guide)
    blended, _ = cg(lambda x: cv2.Laplacian(x, cv2.CV_64F) - lambda_factor * x, b.flatten())
    return blended.reshape(image.shape)


def manipulate_saliency(image, mask, delta_s=0.6, max_iterations=20, patch_size=7):
    """
    Main function to adjust saliency of an image based on user input.
    """
    saliency_map = compute_saliency_map(image)
    if saliency_map is None:
        raise ValueError("Saliency map computation failed.")
    
    tau_positive, tau_negative = 0.5, 0.5  # Initial thresholds
    manipulated_image = image.copy()
    
    for _ in range(max_iterations):
        d_positive, d_negative = extract_saliency_patches(image, saliency_map, tau_positive, tau_negative)
        manipulated_image = patch_match(manipulated_image, mask, d_positive, d_negative, patch_size)
        
        # Update thresholds based on saliency contrast
        saliency_contrast = np.mean(saliency_map[mask > 0]) - np.mean(saliency_map[mask == 0])
        adjustment = np.abs(saliency_contrast - delta_s) * 0.1
        tau_positive += adjustment
        tau_negative -= adjustment
        
        if np.abs(saliency_contrast - delta_s) < 0.05:
            break  # Stop if saliency contrast is close enough
    
        # Apply screened Poisson blending for seamless results
        final_image = solve_screened_poisson(manipulated_image, image)
    return final_image


if __name__ == "__main__":
    input_image = cv2.imread("input.jpg")
    mask = cv2.imread("mask.jpg", cv2.IMREAD_GRAYSCALE)  # Load user-defined mask
    
    result = manipulate_saliency(input_image, mask)
    cv2.imwrite("output.jpg", result)
