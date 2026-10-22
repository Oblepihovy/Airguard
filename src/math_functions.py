import numpy as np

def softplus(a, b, beta=1e4):
    """
    Want to hold potential functions smooth. Actually, don't have to, but can, so why not.
    So, instead of max I will use this one
    """
    m = max(a, b)
    return m + np.log1p(np.exp(-beta * abs(a - b))) / beta