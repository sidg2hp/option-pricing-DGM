"""Evaluation metrics: L2 error, relative error, and Greeks error."""

import numpy as np


def relative_l2_error(
    predicted: np.ndarray, reference: np.ndarray
) -> float:
    r"""Compute the relative L2 error.

    .. math::

        \varepsilon_{L^2} = \frac{\|u_\theta - u_{\text{ref}}\|_2}
                                  {\|u_{\text{ref}}\|_2}

    Parameters
    ----------
    predicted : np.ndarray
        Model predictions.
    reference : np.ndarray
        Reference (analytical or MC) values.

    Returns
    -------
    float
    """
    diff = np.asarray(predicted) - np.asarray(reference)
    ref = np.asarray(reference)
    denom = np.linalg.norm(ref)
    if denom < 1e-15:
        return float(np.linalg.norm(diff))
    return float(np.linalg.norm(diff) / denom)


def max_pointwise_error(
    predicted: np.ndarray, reference: np.ndarray
) -> float:
    """Maximum absolute pointwise error.

    Parameters
    ----------
    predicted, reference : np.ndarray

    Returns
    -------
    float
    """
    return float(np.max(np.abs(np.asarray(predicted) - np.asarray(reference))))


def mean_relative_error(
    predicted: np.ndarray, reference: np.ndarray
) -> float:
    """Mean relative error, ignoring points where reference is near zero.

    Parameters
    ----------
    predicted, reference : np.ndarray

    Returns
    -------
    float
    """
    pred = np.asarray(predicted).ravel()
    ref = np.asarray(reference).ravel()
    mask = np.abs(ref) > 1e-8
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(pred[mask] - ref[mask]) / np.abs(ref[mask])))
