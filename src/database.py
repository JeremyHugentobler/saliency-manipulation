
import numpy as np
import cv2

def compute_image_database(J, D_positve, D_negative, patch_size=7):
    """
    Construct the image representation of the databases.
    """
    # Create D +/- masks based on the saliency of J
    D_positive_mask = np.zeros_like(J)
    D_negative_mask = np.zeros_like(J)

    x,y = D_positve.T
    D_positive_mask[x,y] = 1
    x,y = D_negative.T
    D_negative_mask[x,y] = 1

    # Dilate the masks so that for a pixel (i,j), the whole patch centered at (i,j) is selected
    # D_positive_mask = cv2.dilate(D_positive_mask, np.ones((patch_size, patch_size), np.uint8), iterations=1)
    # D_negative_mask = cv2.dilate(D_negative_mask, np.ones((patch_size, patch_size), np.uint8), iterations=1)

    # Create the database by applying the patch on the image
    D_plus = J * D_positive_mask
    D_minus = J * D_negative_mask
    
    return cv2.cvtColor(D_plus, cv2.COLOR_Lab2RGB), cv2.cvtColor(D_minus, cv2.COLOR_Lab2RGB)

    
def compute_location_database(tau_plus, tau_minus, J, S_J):
    """
    Compute the database of patches D+ and D- as list of coordinates 
    based on the saliency map S_J.

    Args:
        tau_plus: The current value of tau_plus
        J: The current image
        R: The region of interest

    Returns:
        D_plus, D_minus: The database of patches D+ or D-
        D_plus_mask, D_minus_mask: The masks of the DB's
    """
    # Positive DB
    x,y = np.where(S_J > tau_plus)
    D_plus = np.hstack([x[:, None], y[:, None]])

    D_plus_mask = np.zeros_like(S_J)
    D_plus_mask[x,y] = 1

    # Positive DB
    x,y = np.where(S_J < tau_minus)
    D_minus = np.hstack([x[:, None], y[:, None]])

    D_minus_mask = np.zeros_like(S_J)
    D_minus_mask[x,y] = 1



    return D_plus, D_minus, D_plus_mask, D_minus_mask