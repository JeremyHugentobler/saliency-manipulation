import cv2
from scipy.sparse.linalg import cg

def minimize_J(J, I_D_plus, I_D_minus, R):
    """
    Core function that tries to minimize the following energy function:
        E(J,D+,D-) = E+ + E- + E_delta
    where:  E+ = sum{p in R}( min{q in D+} (D(p,q)) )
            E- = sum{p not in R}( min{q in D+} ((D(p,q)) )
            E_delta = ||grad_J - grad_I||
    """

    # For all patches p in R, find closest patch q in D+
    # q = patchmatch(p, D+)

    # Morph q into p using Screened Poisson optimization
    # new_pathch_p = screed_poisson(p, q)
    # j[p_coords] = new_patch_p

    # For all patches p not in R, find closest patch q in D-
    # q = patchmatch(p, D-)

    # Morph q into p using Screened Poisson optimization
    # new_pathch_p = screed_poisson(p, q)
    # j[p_coords] = new_patch_p

    # TODO
    pass

def patchmatch(p, J, D):
    """
    Find the closest patch q in D to p and replace it in the image J.
    """

def screed_poisson(J, J_modified, lambda_factor = 5.0):
    """
    Screened Poisson optimization to smoothly blend the patched image with the original image.

    Args:
        J: The original image
        J_modified: The image with patched regions
        lambda_factor: The lambda factor for the optimization
    Returns:    
        The blended image
    """
    laplacian = cv2.Laplacian(J_modified, cv2.CV_64F)
    b = laplacian - lambda_factor * (J_modified - J)
    blended, _ = cg(lambda x: cv2.Laplacian(x, cv2.CV_64F) - lambda_factor * x, b.flatten())
    return blended.reshape(J.shape)