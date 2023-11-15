import numpy as np

def five_pt_stencil(f, x0, h=1e-4):
    """
    Five-point stencil differentiation based on https://arxiv.org/pdf/2101.00298.pdf

    Parameters:
    -----------

    f : function
        Function to differentiate
    x0 : float, list or np.ndarray
        Reference point at which to compute the derivative of f.
        It can be a float, a list (with len = Ncomponents) or an
        np.ndarray with shape (Ncomponents).

    Returns:
    --------
    der: np.ndarray
        Partial derivative of f with respect to the components of x0
        evaluated at x0, with shape (NComponents, f(x0).shape).
    """
    if (not isinstance(x0, np.ndarray)) and (not isinstance(x0, list)):
        der = 1./(12.*h)*(f(x0 - 2*h) - 8.*f(x0 - h) + 8.*f(x0 + h) - f(x0 + 2*h))
        return der
    elif isinstance(x0, list):
        x0 = np.array(x0)
    if isinstance(x0, np.ndarray):
        if x0.ndim == 1:
            xp2 = x0 + 2.*h*np.identity(len(x0), dtype=np.float64)[:, None]
            xm2 = x0 - 2.*h*np.identity(len(x0), dtype=np.float64)[:, None]
            xp1 = x0 + 1.*h*np.identity(len(x0), dtype=np.float64)[:, None]
            xm1 = x0 - 1.*h*np.identity(len(x0), dtype=np.float64)[:, None]
            xp2 = xp2[:, 0, :]
            xm2 = xm2[:, 0, :]
            xp1 = xp1[:, 0, :]
            xm1 = xm1[:, 0, :]
            der = 1./(12.*h)*(f(xm2) - 8.*f(xm1) + 8.*f(xp1) - f(xp2))
            return der