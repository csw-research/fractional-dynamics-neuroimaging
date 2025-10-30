"""
Long-range dependence (LRD) tests and analysis.

Tests for power-law decay in autocorrelation:
    ρ(k) ~ k^{-γ} where γ = 2 - 2H
"""

import numpy as np
from scipy import stats, signal
from typing import Tuple, Optional


def acf_analysis(
    x: np.ndarray,
    max_lag: Optional[int] = None,
    plot: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute and analyze autocorrelation function.

    For LRD: ACF decays as power law ρ(k) ~ k^{-γ}

    Args:
        x: Time series
        max_lag: Maximum lag (default: min(len(x)//4, 100))
        plot: Whether to create plot

    Returns:
        (lags, acf_values)
    """
    x = np.asarray(x) - np.mean(x)
    n = len(x)

    if max_lag is None:
        max_lag = min(n // 4, 100)

    # Compute ACF
    acf = np.correlate(x, x, mode='full')
    acf = acf[n-1:]  # Take positive lags
    acf = acf / acf[0]  # Normalize

    lags = np.arange(max_lag + 1)
    acf_values = acf[:max_lag + 1]

    if plot:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Linear scale
        ax1.plot(lags, acf_values)
        ax1.set_xlabel('Lag')
        ax1.set_ylabel('ACF')
        ax1.set_title('Autocorrelation Function')
        ax1.grid(True, alpha=0.3)

        # Log-log scale (for power-law check)
        valid = (lags > 0) & (acf_values > 0)
        ax2.loglog(lags[valid], acf_values[valid], 'o-')
        ax2.set_xlabel('Lag (log scale)')
        ax2.set_ylabel('ACF (log scale)')
        ax2.set_title('Log-log ACF (power-law check)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return lags, acf_values


def power_spectrum_analysis(
    x: np.ndarray,
    fs: float = 1.0,
    method: str = "welch"
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Analyze power spectrum for 1/f^β behavior.

    For LRD: S(f) ~ 1/f^β where β = 2H + 1

    Args:
        x: Time series
        fs: Sampling frequency
        method: "periodogram" or "welch"

    Returns:
        (frequencies, psd, spectral_exponent_beta)
    """
    x = np.asarray(x)

    if method == "periodogram":
        freqs, psd = signal.periodogram(x, fs=fs, scaling='density')
    elif method == "welch":
        freqs, psd = signal.welch(x, fs=fs, nperseg=min(256, len(x)//4))
    else:
        raise ValueError(f"Unknown method: {method}")

    # Remove DC and very high frequencies
    valid = (freqs > 0) & (freqs < fs / 4)
    freqs_valid = freqs[valid]
    psd_valid = psd[valid]

    # Fit power law in log-log space
    log_f = np.log(freqs_valid)
    log_psd = np.log(psd_valid)

    beta = -np.polyfit(log_f, log_psd, 1)[0]

    return freqs, psd, beta


def long_range_dependence_test(
    x: np.ndarray,
    method: str = "gph",
    significance: float = 0.05
) -> Tuple[bool, float, dict]:
    """
    Statistical test for long-range dependence.

    Args:
        x: Time series
        method: "gph" (Geweke-Porter-Hudak) or "variance_ratio"
        significance: Significance level

    Returns:
        (has_lrd, test_statistic, info_dict)
    """
    if method == "gph":
        return gph_test(x, significance)
    elif method == "variance_ratio":
        return variance_ratio_test(x, significance)
    else:
        raise ValueError(f"Unknown method: {method}")


def gph_test(
    x: np.ndarray,
    significance: float = 0.05
) -> Tuple[bool, float, dict]:
    """
    Geweke-Porter-Hudak test for LRD.

    Tests whether spectral density near zero frequency
    has power-law form.

    Args:
        x: Time series
        significance: Significance level

    Returns:
        (has_lrd, d_estimate, info)
        where d is the fractional differencing parameter
    """
    n = len(x)

    # Use first m = n^0.5 frequencies
    m = int(n ** 0.5)

    # Compute periodogram
    freqs, psd = signal.periodogram(x, scaling='density')
    freqs = freqs[1:m+1]  # Remove DC
    I = psd[1:m+1]

    # GPH regression: log(I_j) = c - d*log(4*sin²(π*j/n))
    j = np.arange(1, m + 1)
    lambda_j = 2 * np.pi * j / n

    X = np.log(4 * np.sin(lambda_j / 2) ** 2)
    y = np.log(I)

    # OLS regression
    d_hat = -np.polyfit(X, y, 1)[0]

    # Standard error
    se = np.sqrt(np.pi**2 / (6 * m))

    # Test statistic
    z = d_hat / se

    # p-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    has_lrd = p_value < significance and d_hat > 0

    info = {
        "d_estimate": d_hat,
        "std_error": se,
        "z_statistic": z,
        "p_value": p_value,
    }

    return has_lrd, d_hat, info


def variance_ratio_test(
    x: np.ndarray,
    q: int = 10,
    significance: float = 0.05
) -> Tuple[bool, float, dict]:
    """
    Lo-MacKinlay variance ratio test.

    For short-range dependence: VR(q) → 1 as n → ∞
    For long-range dependence: VR(q) grows with q

    Args:
        x: Time series (returns/differences)
        q: Aggregation period
        significance: Significance level

    Returns:
        (has_lrd, vr_statistic, info)
    """
    n = len(x)

    # Variance of 1-period returns
    var_1 = np.var(x, ddof=1)

    # Variance of q-period returns
    x_q = np.sum(x[:n//q * q].reshape(-1, q), axis=1)
    var_q = np.var(x_q, ddof=1)

    # Variance ratio
    vr = var_q / (q * var_1)

    # Under null (no LRD), VR ≈ 1
    # Standard error (homoskedastic)
    se = np.sqrt((2 * (q - 1) * (q - 1)) / (3 * q * n))

    # Test statistic
    z = (vr - 1) / se

    # p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    has_lrd = p_value < significance and vr > 1

    info = {
        "variance_ratio": vr,
        "std_error": se,
        "z_statistic": z,
        "p_value": p_value,
    }

    return has_lrd, vr, info


def estimate_fractional_integration_order(
    x: np.ndarray,
    method: str = "gph"
) -> float:
    """
    Estimate fractional integration order d.

    For ARFIMA(p,d,q) processes:
    - d = 0: Short memory
    - 0 < d < 0.5: Long memory (stationary)
    - d = 0.5: Boundary (non-stationary)

    Args:
        x: Time series
        method: Estimation method

    Returns:
        Estimated d
    """
    if method == "gph":
        _, d, _ = gph_test(x)
        return d
    else:
        raise ValueError(f"Unknown method: {method}")


def memory_parameter_from_hurst(H: float) -> float:
    """
    Convert Hurst exponent to fractional integration parameter.

    d = H - 0.5

    Args:
        H: Hurst exponent

    Returns:
        d: Fractional integration order
    """
    return H - 0.5


def test_whiteness(
    x: np.ndarray,
    max_lag: int = 20
) -> Tuple[bool, float, dict]:
    """
    Ljung-Box test for white noise (no autocorrelation).

    Null hypothesis: Data is white noise (no LRD).

    Args:
        x: Time series
        max_lag: Number of lags to test

    Returns:
        (is_white, statistic, info)
    """
    n = len(x)

    # Compute ACF
    lags, acf = acf_analysis(x, max_lag=max_lag)
    acf = acf[1:]  # Remove lag 0

    # Ljung-Box statistic
    lb_stat = n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, len(acf) + 1)))

    # p-value (chi-squared distribution)
    p_value = 1 - stats.chi2.cdf(lb_stat, df=max_lag)

    is_white = p_value > 0.05

    info = {
        "lb_statistic": lb_stat,
        "p_value": p_value,
        "df": max_lag,
    }

    return is_white, lb_stat, info
