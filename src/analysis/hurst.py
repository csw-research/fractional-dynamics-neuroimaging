"""
Hurst exponent estimation methods.

The Hurst exponent H characterizes long-range dependence:
- H = 0.5: Uncorrelated (Brownian motion)
- H > 0.5: Persistent (positive correlations)
- H < 0.5: Anti-persistent (negative correlations)

Methods:
1. R/S analysis (rescaled range)
2. DFA (detrended fluctuation analysis)
3. Periodogram
4. Wavelet-based
"""

import torch
import numpy as np
from typing import Optional, Tuple
from scipy import signal


def rescaled_range_analysis(
    x: np.ndarray,
    min_window: int = 10,
    max_window: Optional[int] = None,
    num_windows: int = 20
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Estimate Hurst exponent using R/S (rescaled range) analysis.

    Classical method by Hurst (1951).

    Algorithm:
    1. For window size n:
       - Divide series into blocks
       - Compute range R and std S for each block
       - Average R/S across blocks
    2. Plot log(R/S) vs log(n)
    3. Slope ≈ H

    Args:
        x: Time series, shape (N,)
        min_window: Minimum window size
        max_window: Maximum window size (default: N/4)
        num_windows: Number of window sizes to test

    Returns:
        (H, window_sizes, rs_values): Hurst exponent and data for plotting
    """
    x = np.asarray(x)
    n = len(x)

    if max_window is None:
        max_window = n // 4

    # Window sizes (log-spaced)
    window_sizes = np.unique(
        np.logspace(
            np.log10(min_window),
            np.log10(max_window),
            num_windows
        ).astype(int)
    )

    rs_values = []

    for window_size in window_sizes:
        # Split into windows
        num_blocks = n // window_size

        if num_blocks < 1:
            continue

        rs_block = []

        for i in range(num_blocks):
            block = x[i * window_size:(i + 1) * window_size]

            # Mean-adjusted cumulative sum
            mean = np.mean(block)
            y = np.cumsum(block - mean)

            # Range
            R = np.max(y) - np.min(y)

            # Standard deviation
            S = np.std(block, ddof=1)

            if S > 0:
                rs_block.append(R / S)

        if rs_block:
            rs_values.append(np.mean(rs_block))
        else:
            rs_values.append(np.nan)

    # Remove NaN values
    valid = ~np.isnan(rs_values)
    window_sizes = window_sizes[valid]
    rs_values = np.array(rs_values)[valid]

    # Linear regression in log-log space
    log_n = np.log(window_sizes)
    log_rs = np.log(rs_values)

    H = np.polyfit(log_n, log_rs, 1)[0]

    return H, window_sizes, rs_values


def dfa_analysis(
    x: np.ndarray,
    min_window: int = 10,
    max_window: Optional[int] = None,
    num_windows: int = 20,
    order: int = 1
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Detrended Fluctuation Analysis (DFA) for Hurst exponent.

    More robust than R/S analysis, removes local trends.

    Algorithm:
    1. Integrate series: Y(i) = Σ[x(k) - mean(x)]
    2. Divide into windows of size n
    3. Fit polynomial trend in each window
    4. Compute fluctuation F(n) = sqrt(mean(detrended²))
    5. Plot log(F) vs log(n), slope ≈ H

    Args:
        x: Time series
        min_window: Minimum window size
        max_window: Maximum window size
        num_windows: Number of window sizes
        order: Polynomial order for detrending (1=linear, 2=quadratic)

    Returns:
        (H, window_sizes, fluctuations)
    """
    x = np.asarray(x)
    n = len(x)

    if max_window is None:
        max_window = n // 4

    # Integrate (cumulative sum of mean-centered series)
    y = np.cumsum(x - np.mean(x))

    window_sizes = np.unique(
        np.logspace(
            np.log10(min_window),
            np.log10(max_window),
            num_windows
        ).astype(int)
    )

    fluctuations = []

    for window_size in window_sizes:
        num_blocks = n // window_size

        if num_blocks < 1:
            continue

        residuals = []

        for i in range(num_blocks):
            # Extract window
            start = i * window_size
            end = (i + 1) * window_size
            window = y[start:end]

            # Fit polynomial
            t = np.arange(len(window))
            coeffs = np.polyfit(t, window, order)
            trend = np.polyval(coeffs, t)

            # Detrend
            detrended = window - trend

            residuals.extend(detrended)

        # Fluctuation
        F = np.sqrt(np.mean(np.array(residuals) ** 2))
        fluctuations.append(F)

    fluctuations = np.array(fluctuations)

    # Linear regression
    log_n = np.log(window_sizes)
    log_F = np.log(fluctuations)

    H = np.polyfit(log_n, log_F, 1)[0]

    return H, window_sizes, fluctuations


def periodogram_hurst(
    x: np.ndarray,
    fs: float = 1.0
) -> float:
    """
    Estimate Hurst exponent from power spectrum.

    For fractional Brownian motion:
    S(f) ∝ 1/f^β where β = 2H + 1

    Args:
        x: Time series
        fs: Sampling frequency

    Returns:
        Estimated Hurst exponent
    """
    x = np.asarray(x)

    # Compute periodogram
    freqs, psd = signal.periodogram(x, fs=fs, scaling='density')

    # Remove DC component and very high frequencies
    valid = (freqs > 0) & (freqs < fs / 4)
    freqs = freqs[valid]
    psd = psd[valid]

    # Linear regression in log-log space
    log_f = np.log(freqs)
    log_psd = np.log(psd)

    beta = -np.polyfit(log_f, log_psd, 1)[0]  # Negative because S ∝ 1/f^β

    H = (beta - 1) / 2

    # Clamp to valid range
    H = np.clip(H, 0.0, 1.0)

    return H


def estimate_hurst_exponent(
    x: np.ndarray,
    method: str = "dfa",
    **kwargs
) -> float:
    """
    Estimate Hurst exponent using specified method.

    Args:
        x: Time series
        method: "rs" (R/S), "dfa", or "periodogram"
        kwargs: Method-specific arguments

    Returns:
        Estimated Hurst exponent

    Example:
        >>> # Generate fractional Brownian motion
        >>> from src.fractional.special import fractional_brownian_motion
        >>> fbm = fractional_brownian_motion(1000, H=0.7)
        >>> H_est = estimate_hurst_exponent(fbm.numpy(), method="dfa")
        >>> print(f"True H=0.7, Estimated H={H_est:.2f}")
    """
    if method == "rs":
        H, _, _ = rescaled_range_analysis(x, **kwargs)
    elif method == "dfa":
        H, _, _ = dfa_analysis(x, **kwargs)
    elif method == "periodogram":
        H = periodogram_hurst(x, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    return H


def multiscale_hurst(
    x: np.ndarray,
    scales: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate Hurst exponent at multiple scales.

    Useful for detecting non-stationarity or multifractality.

    Args:
        x: Time series
        scales: Array of scales to analyze (default: powers of 2)

    Returns:
        (scales, hurst_values)
    """
    x = np.asarray(x)
    n = len(x)

    if scales is None:
        max_scale = int(np.log2(n / 10))
        scales = 2 ** np.arange(3, max_scale)

    hurst_values = []

    for scale in scales:
        # Coarse-grain the series
        num_blocks = n // scale
        if num_blocks < 10:
            break

        x_coarse = [
            np.mean(x[i * scale:(i + 1) * scale])
            for i in range(num_blocks)
        ]

        # Estimate Hurst
        H = estimate_hurst_exponent(np.array(x_coarse), method="dfa")
        hurst_values.append(H)

    return scales[:len(hurst_values)], np.array(hurst_values)


def hurst_exponent_confidence_interval(
    x: np.ndarray,
    method: str = "dfa",
    num_bootstrap: int = 100,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Estimate Hurst exponent with confidence interval via bootstrap.

    Args:
        x: Time series
        method: Estimation method
        num_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        (H_estimate, lower_bound, upper_bound)
    """
    n = len(x)

    # Original estimate
    H = estimate_hurst_exponent(x, method=method)

    # Bootstrap
    H_bootstrap = []
    for _ in range(num_bootstrap):
        # Resample with replacement
        indices = np.random.randint(0, n, size=n)
        x_boot = x[indices]

        try:
            H_boot = estimate_hurst_exponent(x_boot, method=method)
            H_bootstrap.append(H_boot)
        except:
            continue

    H_bootstrap = np.array(H_bootstrap)

    # Confidence interval
    alpha = 1 - confidence
    lower = np.percentile(H_bootstrap, 100 * alpha / 2)
    upper = np.percentile(H_bootstrap, 100 * (1 - alpha / 2))

    return H, lower, upper
