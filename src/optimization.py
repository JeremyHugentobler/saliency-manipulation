

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

    
    pass

def screed_poisson(p, q):
    """
    Screened Poisson optimization to morph q into p
    """
    pass