import numpy as np
import logging

logger = logging.getLogger(__name__)


def five_pt_stencil(f, x0, h=1e-4):
    """
    Five-point stencil differentiation based on https://arxiv.org/pdf/2101.00298.pdf

    The routine is designed for internal use only and is not guaranteed to be robust for all inputs.
    It is the user's responsibility to ensure that the step size h is appropriate for the function
    f and the point x0 at which the derivative is being evaluated.

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
    # Input validation
    if not callable(f):
        raise ValueError("f must be callable")

    if not isinstance(h, (int, float)) or h <= 0:
        raise ValueError("h must be a positive number")

    if h < 1e-12:
        logger.warning("Step size h=%g is very small, may cause numerical instability", h)

    # Convert x0 to numpy array if needed
    if isinstance(x0, list):
        x0 = np.array(x0, dtype=np.float64)
    elif not isinstance(x0, np.ndarray):
        x0 = np.array([x0], dtype=np.float64)

    if not np.isfinite(x0).all():
        raise ValueError("x0 contains non-finite values")

    try:
        # Evaluate function at central point first
        f0 = f(x0)
        if not np.isfinite(f0).all():
            logger.warning("Function evaluation at x0 returns non-finite values: %s", f0)
    except Exception as e:
        raise RuntimeError(f"Function evaluation failed at x0: {e}")

    if (not isinstance(x0, np.ndarray)) and len(x0) == 1:
        # Scalar case
        try:
            fm2 = f(x0 - 2*h)
            fm1 = f(x0 - h)
            fp1 = f(x0 + h)
            fp2 = f(x0 + 2*h)

            # Check for non-finite values
            for val, label in [(fm2, 'x0-2h'), (fm1, 'x0-h'), (fp1, 'x0+h'), (fp2, 'x0+2h')]:
                if not np.isfinite(val).all():
                    logger.warning("Function evaluation at %s returns non-finite values: %s",
                                   label, val)

            der = 1./(12.*h)*(fm2 - 8.*fm1 + 8.*fp1 - fp2)

            if not np.isfinite(der).all():
                logger.warning("Derivative calculation produced non-finite values: %s", der)

            return der

        except Exception as e:
            raise RuntimeError(f"Function evaluation failed during differentiation: {e}")

    elif isinstance(x0, np.ndarray):
        if x0.ndim == 1:
            try:
                # Vector case - create perturbation matrices
                n_components = len(x0)
                xp2 = x0 + 2.*h*np.identity(n_components, dtype=np.float64)[:, None]
                xm2 = x0 - 2.*h*np.identity(n_components, dtype=np.float64)[:, None]
                xp1 = x0 + 1.*h*np.identity(n_components, dtype=np.float64)[:, None]
                xm1 = x0 - 1.*h*np.identity(n_components, dtype=np.float64)[:, None]

                xp2 = xp2[:, 0, :]
                xm2 = xm2[:, 0, :]
                xp1 = xp1[:, 0, :]
                xm1 = xm1[:, 0, :]

                # Evaluate function at all points
                fm2_vals = f(xm2)
                fm1_vals = f(xm1)
                fp1_vals = f(xp1)
                fp2_vals = f(xp2)

                # Check for non-finite values in function evaluations
                for vals, label in [(fm2_vals, 'x-2h'), (fm1_vals, 'x-h'),
                                    (fp1_vals, 'x+h'), (fp2_vals, 'x+2h')]:
                    if not np.isfinite(vals).all():
                        logger.warning("Function evaluations at %s contain non-finite values",
                                       label)

                der = 1./(12.*h)*(fm2_vals - 8.*fm1_vals + 8.*fp1_vals - fp2_vals)

                if not np.isfinite(der).all():
                    logger.warning("Derivative calculation produced non-finite values")

                return der

            except Exception as e:
                raise RuntimeError(f"Function evaluation failed during vector differentiation: {e}")
        else:
            raise ValueError("x0 must be 1-dimensional")
    else:
        raise ValueError("Unsupported x0 type")
