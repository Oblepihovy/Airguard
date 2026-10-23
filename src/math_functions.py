import numpy as np

def softplus(a, b, beta=1e4):
    """
    Want to hold potential functions smooth. Actually, don't have to (for FD), but can, so why not.
    So, instead of max I will use this one
    """
    return np.maximum(a, b) + np.log1p(np.exp(-beta * abs(a - b))) / beta