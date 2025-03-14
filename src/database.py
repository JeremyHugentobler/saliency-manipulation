
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

    D_plus = J * (S_J > tau_plus)
    D_minus = J * (S_J < tau_minus)

    D_plus = cv2.dilate(D_plus, np.ones((patch_size, patch_size), np.uint8), iterations=1)
    D_minus = cv2.dilate(D_minus, np.ones((patch_size, patch_size), np.uint8), iterations=1)

    return D_plus, D_minus