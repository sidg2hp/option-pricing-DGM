"""Mathematical utilities: correlation matrix validation and Cholesky factorisation.

Provides helper functions used throughout the codebase for constructing
and validating the diffusion structure of multi-asset models.
"""

from typing import Tuple

import numpy as np
import torch


def validate_correlation_matrix(rho: np.ndarray) -> np.ndarray:
    """Validate a correlation matrix and return its Cholesky factor.

    Checks that *rho* is symmetric, has unit diagonal, and is positive
    definite.  Returns the lower-triangular Cholesky factor
    ``L`` such that ``L @ L.T == rho``.

    Parameters
    ----------
    rho : np.ndarray
        Correlation matrix of shape ``(d, d)``.

    Returns
    -------
    np.ndarray
        Lower-triangular Cholesky factor of shape ``(d, d)``.

    Raises
    ------
    ValueError
        If *rho* is not symmetric, does not have unit diagonal,
        or is not positive definite.
    """
    rho = np.asarray(rho, dtype=np.float64)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError(f"Correlation matrix must be square, got shape {rho.shape}")
    if not np.allclose(rho, rho.T, atol=1e-8):
        raise ValueError(
            "Correlation matrix is not symmetric. "
            f"Max asymmetry: {np.max(np.abs(rho - rho.T)):.2e}"
        )
    if not np.allclose(np.diag(rho), 1.0, atol=1e-8):
        raise ValueError(
            f"Diagonal entries must be 1. Got diag = {np.diag(rho)}"
        )
    try:
        L = np.linalg.cholesky(rho)
    except np.linalg.LinAlgError:
        eigvals = np.linalg.eigvalsh(rho)
        raise ValueError(
            f"Correlation matrix is not positive definite. "
            f"Eigenvalues: {eigvals}"
        )
    return L


def build_diffusion_tensor(
    sigma: np.ndarray, rho: np.ndarray
) -> np.ndarray:
    """Build the diffusion tensor A[i,j] = rho[i,j] * sigma[i] * sigma[j].

    Parameters
    ----------
    sigma : np.ndarray
        Volatility vector of shape ``(d,)``.
    rho : np.ndarray
        Correlation matrix of shape ``(d, d)``.

    Returns
    -------
    np.ndarray
        Diffusion tensor of shape ``(d, d)``.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    rho = np.asarray(rho, dtype=np.float64)
    return rho * np.outer(sigma, sigma)


def build_equicorrelation_matrix(d: int, rho_off: float) -> np.ndarray:
    """Build an equicorrelation matrix with off-diagonal value *rho_off*.

    Parameters
    ----------
    d : int
        Dimension.
    rho_off : float
        Off-diagonal correlation value.  Must satisfy
        ``rho_off > -1/(d-1)`` for the matrix to be positive definite.

    Returns
    -------
    np.ndarray
        Correlation matrix of shape ``(d, d)``.
    """
    mat = np.full((d, d), rho_off, dtype=np.float64)
    np.fill_diagonal(mat, 1.0)
    return mat
