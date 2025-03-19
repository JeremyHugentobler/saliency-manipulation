
import numpy as np
import cv2

def compute_database(tau_plus, tau_minus, J, S_J, patch_size=5):
    """
    Compute the database of patches D+ and D- based on the current image J.

    Args:
        tau_plus: The current value of tau_plus
        J: The current image
        R: The region of interest

    Returns:
        The database of patches D+ or D-
    """
    D_plus = np.zeros_like(J)
    D_minus = np.zeros_like(J)

    # Create D +/- masks based on the saliency of J
    D_plus_mask = (S_J > tau_plus) * 1
    D_minus_mask = (S_J < tau_minus) * 1

    # Dilate the masks so that for a pixel (i,j), the whole patch centered at (i,j) is selected
    D_plus_mask = cv2.dilate(D_plus_mask, np.ones((patch_size, patch_size), np.uint8), iterations=1)
    D_minus_mask = cv2.dilate(D_minus_mask, np.ones((patch_size, patch_size), np.uint8), iterations=1)

    # Create the database by applying the patch on the image
    D_plus = J * D_plus_mask
    D_minus = J * D_minus_mask

    return D_plus, D_minus