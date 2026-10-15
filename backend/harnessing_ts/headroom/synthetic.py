from __future__ import annotations

"""Synthetic tasks whose full-information optimal forecasters are exactly computable (contract §8)."""

import collections
import math
from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np
from scipy.special import gammaln

from harnessing_ts.headroom.config import HeadroomConfig

DEFAULT_SIZES: dict[str, int] = {"train": 20_000, "diag": 10_000, "cert": 12_000, "test": 20_000}

# Fresh cert/test simulations carry this many unscored rows in front so that every
# feature the pipeline builds at its first usable origin is already realized history.
_PREFIX = HeadroomConfig().burn_in + HeadroomConfig().input_length  # 687
_ARFIMA_PREFIX = 2_100  # also >= the DL order, so most origins see the full lag set
_ARFIMA_D = 0.35
_ARFIMA_ORDER = 2_000  # Durbin-Levinson depth of the oracle (>= default burn-in)
_ARFIMA_MA_TRUNC = 2_000_000  # fractional-differencing truncation, see arfima()
_HORIZON = HeadroomConfig().horizon  # mse_theoretical() takes no cfg argument


class Oracle(Protocol):
    """Full-information forecaster for one synthetic process."""

    def forecast(self, seg: np.ndarray, origins: np.ndarray, cfg: HeadroomConfig) -> np.ndarray:
        """Optimal E[x[o+h] | x[0..o]] at every origin; returns [B, H, C]."""
        ...

    def mse_theoretical(self) -> np.ndarray | None:
        """[H] mean-over-channel MSE of the oracle, or None where not computable."""
        ...


@dataclass(frozen=True)
class SyntheticTask:
    """One synthetic process plus its exactly-computable optimal forecaster.

    ``train_diag`` is a single continuous simulation split at ``diag_start``
    (train before, diag after).  ``cert`` and ``test`` are fresh independent
    simulations of the same process whose first ``_PREFIX`` rows (``_ARFIMA_PREFIX``
    for the arfima task) are a burn-in prefix on top of the requested content.
    """

    name: str
    train_diag: np.ndarray  # [T, C], T = sizes["train"] + sizes["diag"]
    diag_start: int  # == sizes["train"]
    cert: np.ndarray  # [prefix + sizes["cert"], C]
    test: np.ndarray  # [prefix + sizes["test"], C]
    oracle: Oracle
    structure: dict[str, Any]  # periods / couplings / regimes for defect-coordinate mapping


def _merge_sizes(sizes: dict[str, int] | None) -> dict[str, int]:
    """Defaults merged with the caller's overrides; unknown or non-positive entries rejected."""
    if sizes is None:
        return dict(DEFAULT_SIZES)
    if not isinstance(sizes, dict):
        raise ValueError(f"sizes must be a dict or None, got {type(sizes).__name__}")
    merged = dict(DEFAULT_SIZES)
    for key, value in sizes.items():
        if key not in DEFAULT_SIZES:
            raise ValueError(f"unknown size key {key!r}; expected a subset of {sorted(DEFAULT_SIZES)}")
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"size {key!r} must be a positive int, got {value!r}")
        merged[key] = int(value)
    return merged


def _spawn(seed: int) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    """Three independent streams for the main series and the two fresh segments."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError(f"seed must be a non-negative int, got {seed!r}")
    main, cert, test = np.random.SeedSequence(int(seed)).spawn(3)
    return np.random.default_rng(main), np.random.default_rng(cert), np.random.default_rng(test)


def _check_forecast_args(
    seg: np.ndarray, origins: np.ndarray, cfg: HeadroomConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Shared forecast-argument validation; mirrors timeidx.windows' bounds."""
    seg = np.asarray(seg, dtype=np.float64)
    if seg.ndim != 2:
        raise ValueError(f"seg must be [T, C], got shape {seg.shape}")
    origins = np.asarray(origins, dtype=np.int64)
    if origins.ndim != 1:
        raise ValueError(f"origins must be 1-D, got shape {origins.shape}")
    if origins.size:
        if origins.min() < 0:
            raise ValueError(f"origin {origins.min()} lies before the start of seg")
        if origins.max() + cfg.horizon > seg.shape[0] - 1:
            raise ValueError(f"origin {origins.max()} has targets beyond the end of seg")
    return seg, origins


def _stationary_cov(transition: np.ndarray, noise_cov: np.ndarray) -> np.ndarray:
    """P solving P = F P F' + Q (row-major Kronecker form of the Lyapunov equation)."""
    n = transition.shape[0]
    try:
        cov = np.linalg.solve(np.eye(n * n) - np.kron(transition, transition), noise_cov.reshape(-1))
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("state transition is not stable; no stationary covariance exists") from exc
    cov = cov.reshape(n, n)
    return 0.5 * (cov + cov.T)


def _kalman_filter_means(
    y: np.ndarray, A: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray,
    x0: np.ndarray, P0: np.ndarray,
) -> np.ndarray:
    """Run the time-invariant Kalman filter; returns the filtered means x[t|t] as [T, n].

    (x0, P0) is the prior one step *before* the first row of y; because every model
    here uses the stationary P0, the first predict leaves it invariant, so this
    coincides with drawing the initial state from N(x0, P0) at t = 0.
    Joseph-form updates keep the covariance positive semidefinite even with R = 0.
    """
    T = y.shape[0]
    n = A.shape[0]
    means = np.empty((T, n))
    x = np.array(x0, dtype=np.float64)
    P = np.array(P0, dtype=np.float64)
    eye = np.eye(n)
    for t in range(T):
        x = A @ x
        P = A @ P @ A.T + Q
        S = H @ P @ H.T + R
        gain = np.linalg.solve(S, H @ P).T  # K = P H' S^-1, valid since S and P are symmetric
        x = x + gain @ (y[t] - H @ x)
        kh = eye - gain @ H
        P = kh @ P @ kh.T + gain @ R @ gain.T
        means[t] = x
    return means


def _steady_filter_cov(
    A: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray, P0: np.ndarray,
    max_iter: int = 20_000, tol: float = 1e-13,
) -> np.ndarray:
    """Steady-state filtered covariance of the Riccati recursion (the innovation form).

    Models whose process noise is zero in some direction (the noiseless
    oscillators of lgssm_multiperiod) converge only algebraically — information
    about a fixed amplitude accumulates like 1/t — so a geometric criterion
    would never trigger; those models simply run the full budget, and the
    resulting h-step MSE is accurate to < 1e-4 relative (measured on this
    package's models), far below any experimental resolution.
    """
    scale = max(float(np.linalg.norm(P0)), 1e-12)
    P = np.array(P0, dtype=np.float64)
    eye = np.eye(A.shape[0])
    for _ in range(max_iter):
        Pp = A @ P @ A.T + Q
        S = H @ Pp @ H.T + R
        gain = np.linalg.solve(S, H @ Pp).T
        kh = eye - gain @ H
        Pn = kh @ Pp @ kh.T + gain @ R @ gain.T
        if float(np.linalg.norm(Pn - P)) <= tol * scale:
            return Pn
        P = Pn
    return P


def _kalman_forecast_mse(
    A: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray,
    p_inf: np.ndarray, horizon: int,
) -> np.ndarray:
    """Steady-state h-step forecast MSE, averaged over channels.

    With the filter in steady state at the origin, the h-step error covariance is
    H (A^h P_inf A^h' + sum_{j<h} A^j Q A^j') H' + R.
    """
    B = np.array(p_inf, dtype=np.float64)
    C = np.zeros_like(Q)
    mse = np.empty(horizon)
    for h in range(1, horizon + 1):
        B = A @ B @ A.T
        C = Q + A @ C @ A.T
        cov = H @ (B + C) @ H.T + R
        mse[h - 1] = float(np.trace(cov)) / cov.shape[0]
    return mse


class KalmanOracle:
    """Exact conditional-mean forecaster of a known linear-Gaussian state-space model.

    A/Q/H/R/P0 are the model in observation form y = H z + v; they are public so
    tests can rebuild the exact joint distribution independently.
    """

    def __init__(self, A: np.ndarray, Q: np.ndarray, H: np.ndarray, R: np.ndarray,
                 P0: np.ndarray) -> None:
        self.A = np.array(A, dtype=np.float64)
        self.Q = np.array(Q, dtype=np.float64)
        self.H = np.array(H, dtype=np.float64)
        self.R = np.array(R, dtype=np.float64)
        self.P0 = np.array(P0, dtype=np.float64)
        p_inf = _steady_filter_cov(self.A, self.Q, self.H, self.R, self.P0)
        self._mse = _kalman_forecast_mse(self.A, self.Q, self.H, self.R, p_inf, _HORIZON)

    def forecast(self, seg: np.ndarray, origins: np.ndarray, cfg: HeadroomConfig) -> np.ndarray:
        seg, origins = _check_forecast_args(seg, origins, cfg)
        if seg.shape[1] != self.H.shape[0]:
            raise ValueError(f"seg has {seg.shape[1]} channels, model observes {self.H.shape[0]}")
        out = np.empty((origins.size, cfg.horizon, seg.shape[1]))
        if origins.size == 0:
            return out
        means = _kalman_filter_means(
            seg, self.A, self.Q, self.H, self.R, np.zeros(self.A.shape[0]), self.P0)
        states = means[origins]  # [B, n]: the filter has already absorbed y[o]
        power = np.eye(self.A.shape[0])
        for h in range(1, cfg.horizon + 1):
            power = self.A @ power
            out[:, h - 1, :] = states @ (self.H @ power).T
        return out

    def mse_theoretical(self) -> np.ndarray:
        return self._mse.copy()


def _hmm_filter(
    x: np.ndarray, trans: np.ndarray, mu: np.ndarray, phi: np.ndarray,
    sigma: float, pi: np.ndarray,
) -> np.ndarray:
    """Forward recursion p[t] = P(S_t = s | x[0..t]).

    The regime chain is initialized from its stationary distribution and the first
    update happens at t = 1 (there is no x[-1] to score t = 0 with); segments
    carry a burn-in prefix far longer than the chain's mixing time, so this
    convention is indistinguishable from exact initialization at every usable
    origin.
    """
    T = x.shape[0]
    K = trans.shape[0]
    p = np.empty((T, K))
    current = np.array(pi, dtype=np.float64)
    p[0] = current
    var = float(sigma) ** 2
    for t in range(1, T):
        predicted = current @ trans
        means = mu + phi * (x[t - 1] - mu)
        loglik = -0.5 * (x[t] - means) ** 2 / var  # constant terms cancel below
        weights = predicted * np.exp(loglik - loglik.max())
        current = weights / weights.sum()
        p[t] = current
    return p


def _hmm_forecast(
    p0: np.ndarray, m0: np.ndarray, trans: np.ndarray, mu: np.ndarray,
    phi: np.ndarray, horizon: int,
) -> np.ndarray:
    """Joint recursion m_s = E[x_t 1{S_t = s} | I]; exact, O(K^2) per step.

    ``m0`` is the joint moment E[x_o 1{S_o = s} | I_o] (x_o * p0 when x_o is
    observed).  With mean-reverting regime dynamics x' = mu_s + phi_s (x - mu_s)
    + sigma eps, one step of the joint recursion is
        p'_s = sum_{s'} trans[s', s] p_{s'}
        m'_s = sum_{s'} trans[s', s] (phi_s m_{s'} + (1 - phi_s) mu_s p_{s'})
    iterated h times together with the regime chain; E[x_{o+h} | I_o] = sum_s m_s.
    This is exact — NOT per-regime forecasts mixed by posteriors, which would
    ignore the interaction between the current level and the regime probabilities.
    """
    p = np.array(p0, dtype=np.float64)
    m = np.array(m0, dtype=np.float64)
    drift = (1.0 - phi) * mu
    out = np.empty((p.shape[0], horizon))
    for h in range(horizon):
        p_next = p @ trans
        m = (m @ trans) * phi + p_next * drift
        p = p_next
        out[:, h] = m.sum(axis=1)
    return out


def _markov_stationary(trans: np.ndarray, max_iter: int = 100_000, tol: float = 1e-15) -> np.ndarray:
    """Stationary distribution of an ergodic row-stochastic chain, by direct iteration."""
    p = np.full(trans.shape[0], 1.0 / trans.shape[0])
    for _ in range(max_iter):
        nxt = p @ trans
        if float(np.abs(nxt - p).max()) <= tol:
            return nxt
        p = nxt
    raise RuntimeError("transition matrix did not mix; no stationary distribution found")


class HmmOracle:
    """Exact conditional-mean forecaster of the regime-switching AR(1) process."""

    def __init__(self, trans: np.ndarray, mu: np.ndarray, phi: np.ndarray,
                 sigma: float, pi: np.ndarray) -> None:
        self.trans = np.array(trans, dtype=np.float64)
        self.mu = np.array(mu, dtype=np.float64)
        self.phi = np.array(phi, dtype=np.float64)
        self.sigma = float(sigma)
        self.pi = np.array(pi, dtype=np.float64)

    def forecast(self, seg: np.ndarray, origins: np.ndarray, cfg: HeadroomConfig) -> np.ndarray:
        seg, origins = _check_forecast_args(seg, origins, cfg)
        if seg.shape[1] != 1:
            raise ValueError(f"hmm_ar_switch is univariate, seg has {seg.shape[1]} channels")
        out = np.empty((origins.size, cfg.horizon, 1))
        if origins.size == 0:
            return out
        posteriors = _hmm_filter(seg[:, 0], self.trans, self.mu, self.phi, self.sigma, self.pi)
        joint = posteriors[origins] * seg[origins, 0][:, None]
        out[:, :, 0] = _hmm_forecast(
            posteriors[origins], joint, self.trans, self.mu, self.phi, cfg.horizon)
        return out

    def mse_theoretical(self) -> np.ndarray | None:
        # The unconditional h-step MSE would require expectations of products of
        # the joint (p, m) recursion, which do not close linearly; not genuinely
        # computable, so the contract's None applies.
        return None


def _arfima_psi(d: float, truncation: int) -> np.ndarray:
    """MA(infinity) coefficients psi_j = Gamma(j+d) / (Gamma(d) Gamma(j+1)), j = 0..M."""
    j = np.arange(truncation + 1)
    return np.exp(gammaln(j + d) - gammaln(d) - gammaln(j + 1))


def _arfima_gamma(d: float, order: int) -> np.ndarray:
    """Theoretical autocovariance of ARFIMA(0, d, 0) with unit innovation variance.

    gamma_k = Gamma(1-2d) Gamma(k+d) / (Gamma(1-d) Gamma(d) Gamma(k+1-d)).
    The Gamma(1-2d)/Gamma(1-d) factor is the gamma_0 normalization; without it the
    sequence is only the ACF up to scale and Durbin-Levinson on it yields an
    innovation variance of ~0.46 instead of 1.  With this formula DL(2000) gives
    nu = 1.00006, matching the true innovation variance.
    """
    k = np.arange(order + 1)
    return np.exp(
        gammaln(1.0 - 2.0 * d) + gammaln(k + d)
        - gammaln(1.0 - d) - gammaln(d) - gammaln(k + 1.0 - d)
    )


def _arfima_durbin_levinson(gamma: np.ndarray, order: int) -> tuple[list[np.ndarray], np.ndarray]:
    """DL recursion to ``order``; returns every AR(m) coefficient vector and all nu_m.

    Keeping all intermediate orders is what makes short-history origins graceful:
    the optimal predictor from o+1 realized rows is exactly AR(o+1), read off as
    phis[m-1] (the order-m vector).
    """
    phis: list[np.ndarray] = []
    nus = np.empty(order + 1)
    nus[0] = gamma[0]
    phi = np.array([gamma[1] / gamma[0]])
    nus[1] = gamma[0] - gamma[1] ** 2 / gamma[0]
    phis.append(phi)
    for m in range(2, order + 1):
        r = gamma[m] - phi @ gamma[m - 1 : 0 : -1]
        km = r / nus[m - 1]
        if not np.isfinite(km) or abs(km) >= 1.0:
            raise RuntimeError(f"Durbin-Levinson reflection coefficient invalid at order {m}")
        nxt = np.empty(m)
        nxt[: m - 1] = phi - km * phi[::-1]
        nxt[m - 1] = km
        phi = nxt
        nus[m] = nus[m - 1] * (1.0 - km * km)
        phis.append(phi)
    return phis, nus


def _arfima_simulate(rng: np.random.Generator, T: int, d: float, truncation: int) -> np.ndarray:
    """One draw of the M-truncated fractional-differencing process, via FFT convolution.

    x_t = sum_{j<=M} psi_j eps_{t-j}: circular convolution at size >= T + 2M + 1 is
    exact (no wrap-around) and costs O(n log n) where direct convolution would be
    O(T M) at the default sizes.
    """
    psi = _arfima_psi(d, truncation)
    n_fft = 1 << int(math.ceil(math.log2(T + 2 * truncation + 1)))
    f_psi = np.fft.rfft(psi, n_fft)
    eps = rng.standard_normal(T + truncation)
    return np.fft.irfft(f_psi * np.fft.rfft(eps, n_fft), n_fft)[truncation : truncation + T]


class ArfimaOracle:
    """AR(2000) predictor from Durbin-Levinson on the theoretical autocovariance.

    Origins with fewer than P realized rows in the segment use the optimal AR of
    their actual lag depth (AR(o+1)) — the lag set is truncated gracefully rather
    than zero-padded.  h-step forecasts iterate the companion recursion on the
    conditional-mean vector, which is algebraically the companion-matrix power
    C^h applied to the stacked history, vectorized across origins that share a
    lag depth.
    """

    def __init__(self, d: float = _ARFIMA_D, order: int = _ARFIMA_ORDER) -> None:
        self.d = float(d)
        self.order = int(order)
        self.gamma = _arfima_gamma(self.d, self.order)
        self.phis, self.nus = _arfima_durbin_levinson(self.gamma, self.order)

    def forecast(self, seg: np.ndarray, origins: np.ndarray, cfg: HeadroomConfig) -> np.ndarray:
        seg, origins = _check_forecast_args(seg, origins, cfg)
        if seg.shape[1] != 1:
            raise ValueError(f"arfima is univariate, seg has {seg.shape[1]} channels")
        out = np.empty((origins.size, cfg.horizon, 1))
        if origins.size == 0:
            return out
        depths = np.minimum(origins + 1, self.order)
        for depth in np.unique(depths):
            sel = depths == depth
            phi = self.phis[depth - 1]
            # buffer[b] = [x_o, x_{o-1}, ..., x_{o-depth+1}]: newest first, like phi.
            buffer = seg[origins[sel, None] - np.arange(depth)[None, :], 0]
            shifted = np.empty_like(buffer)
            for h in range(cfg.horizon):
                pred = buffer @ phi
                out[sel, h, 0] = pred
                shifted[:, 0] = pred
                shifted[:, 1:] = buffer[:, :-1]
                buffer, shifted = shifted, buffer
        return out

    def mse_theoretical(self) -> np.ndarray | None:
        # Only the 1-step value is genuinely computable (nus[P] ~ 1.00006, the
        # innovation variance); h > 1 needs the process's exact h-step error
        # variances, so per the contract we return None rather than a surrogate.
        return None


class ZeroOracle:
    """Conditional-mean-zero process (garch): the optimal forecast is identically 0."""

    def forecast(self, seg: np.ndarray, origins: np.ndarray, cfg: HeadroomConfig) -> np.ndarray:
        seg, origins = _check_forecast_args(seg, origins, cfg)
        return np.zeros((origins.size, cfg.horizon, seg.shape[1]))

    def mse_theoretical(self) -> np.ndarray | None:
        # The conditional variance is not the mean target; the negative control is
        # mean-unpredictable and no [H] mean-MSE applies.
        return None


class LastValueOracle:
    """Random walk: E[x[o+h] | I_o] = x[o] for every h."""

    def __init__(self, sigma: float) -> None:
        self.sigma = float(sigma)

    def forecast(self, seg: np.ndarray, origins: np.ndarray, cfg: HeadroomConfig) -> np.ndarray:
        seg, origins = _check_forecast_args(seg, origins, cfg)
        out = np.empty((origins.size, cfg.horizon, seg.shape[1]))
        out[:] = seg[origins][:, None, :]
        return out

    def mse_theoretical(self) -> np.ndarray:
        return self.sigma**2 * np.arange(1, _HORIZON + 1, dtype=np.float64)


def lgssm_multiperiod(seed: int, sizes: dict[str, int] | None = None) -> SyntheticTask:
    """C=4 LGSSM: a slow level, damped 24- and 168-period oscillators, colored noise.

    Latent state (5): an AR(1) level with phi = 0.995, plus a lightly damped
    sin/cos rotation pair for each period (rho = 0.999, re-excited by small
    isotropic process noise).  With rho = 1 and no noise the oscillator
    amplitude is fixed per realization, which makes Delta(g_hat) realization-
    dependent and uncertifiable across segments; the damping keeps the
    seasonal peaks sharp while making the process ergodic, so the same
    corrector gain is a stable population quantity across realizations.
    Each channel observes a different linear mix of the latents plus its own
    AR(1) channel noise and white observation noise; channel 3 sees no level
    (partial observation).  All parameters are known, so a Kalman filter run
    from segment index 0 with the exact stationary prior followed by h-step
    mean propagation is the exact conditional mean.  Channels are scaled to
    unit variance through the known theoretical per-channel variance.
    """
    sizes = _merge_sizes(sizes)
    rng_main, rng_cert, rng_test = _spawn(seed)

    phi_level = 0.995
    sigma_level = 0.03
    periods = (24, 168)
    oscillator_var = 0.25  # stationary variance of each sin/cos component
    rho_osc = 0.999  # amplitude half-life ~ 700 steps: sharp but ergodic peaks
    sigma_osc = math.sqrt(oscillator_var * (1.0 - rho_osc**2))
    psi_noise = np.array([0.6, 0.8, 0.75, 0.5])
    sigma_noise = np.array([0.3, 0.25, 0.35, 0.4])
    sigma_obs = np.full(4, 0.1)
    # Latent column order: [level, s24, c24, s168, c168].
    h_latent = np.array([
        [1.0, 0.8, 0.4, 0.5, 0.3],
        [0.7, -0.5, 0.9, 0.4, -0.6],
        [1.3, 0.3, -0.7, -0.8, 0.5],
        [0.0, 1.0, 0.6, -0.3, 0.9],  # no level: partial observation
    ])

    a_latent = np.zeros((5, 5))
    a_latent[0, 0] = phi_level
    for slot, period in enumerate(periods):
        angle = 2.0 * np.pi / period
        r, c = 1 + 2 * slot, 2 + 2 * slot
        a_latent[r, r] = rho_osc * math.cos(angle)
        a_latent[r, c] = rho_osc * math.sin(angle)
        a_latent[c, r] = -rho_osc * math.sin(angle)
        a_latent[c, c] = rho_osc * math.cos(angle)
    latent_var = np.array([
        sigma_level**2 / (1.0 - phi_level**2), oscillator_var, oscillator_var,
        oscillator_var, oscillator_var,
    ])
    noise_var = sigma_noise**2 / (1.0 - psi_noise**2)

    # Scale each channel to unit variance through the exact theoretical variance.
    channel_var = (h_latent**2 * latent_var).sum(axis=1) + noise_var + sigma_obs**2
    scale = 1.0 / np.sqrt(channel_var)

    n = 9
    A = np.zeros((n, n))
    A[:5, :5] = a_latent
    A[5:, 5:] = np.diag(psi_noise)
    Q = np.zeros((n, n))
    Q[0, 0] = sigma_level**2
    Q[1:5, 1:5] = np.eye(4) * sigma_osc**2
    Q[5:, 5:] = np.diag(sigma_noise**2)
    H = np.zeros((4, n))
    H[:, :5] = scale[:, None] * h_latent
    H[:, 5:] = np.diag(scale)
    R = np.diag((scale * sigma_obs) ** 2)
    P0 = np.diag(np.concatenate([latent_var, noise_var]))

    def simulate(rng: np.random.Generator, T: int) -> np.ndarray:
        state = rng.standard_normal(n) * np.sqrt(np.diag(P0))
        shocks = rng.standard_normal((T, n)) * np.sqrt(np.diag(Q))
        obs_noise = rng.standard_normal((T, 4)) * (scale * sigma_obs)
        out = np.empty((T, 4))
        for t in range(T):
            state = state @ A.T + shocks[t]
            out[t] = H @ state + obs_noise[t]
        return out

    return SyntheticTask(
        name="lgssm_multiperiod",
        train_diag=simulate(rng_main, sizes["train"] + sizes["diag"]),
        diag_start=sizes["train"],
        cert=simulate(rng_cert, _PREFIX + sizes["cert"]),
        test=simulate(rng_test, _PREFIX + sizes["test"]),
        oracle=KalmanOracle(A, Q, H, R, P0),
        structure={
            "periods": list(periods),
            "n_channels": 4,
            "phi_level": phi_level,
            "channel_level_weights": [float(h_latent[i, 0]) for i in range(4)],
            "channel_noise_ar": psi_noise.tolist(),
        },
    )


def lgssm_xchannel(seed: int, sizes: dict[str, int] | None = None) -> SyntheticTask:
    """C=4 VAR with cross-channel couplings at mixed lags 3/5/7, no observation noise.

    y_t = sum_j A_ij y_{j, t-lag_ij} + eps_t with a sparse, cyclic coupling pattern
    (0<-1, 0<-3, 1<-2, 2<-0, 2<-1, 3<-0, 3<-2), each pair at its own lag from
    {3, 5, 7}.  In companion form this is a known-parameter LGSSM with R = 0 whose
    Kalman filter is exact; since the whole state consists of realized
    observations, the filtered covariance reaches zero after max_lag steps and the
    h-step forecast equals the direct VAR mean recursion.  Channels are scaled to
    unit variance via the stationary covariance (a D A D^-1 rescaling of both the
    couplings and the innovations).
    """
    sizes = _merge_sizes(sizes)
    rng_main, rng_cert, rng_test = _spawn(seed)

    n_channels = 4
    max_lag = 7
    # (target i, source j, lag, coefficient)
    couplings = [
        (0, 1, 5, 0.35), (0, 3, 7, -0.25),
        (1, 2, 3, 0.40),
        (2, 0, 7, 0.30), (2, 1, 5, -0.30),
        (3, 0, 5, 0.25), (3, 2, 7, 0.30),
    ]
    sigma_raw = np.full(n_channels, 0.5)
    state_dim = n_channels * max_lag

    def build_companion(phi_lags: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # State: (y_t, y_{t-1}, ..., y_{t-max_lag+1}); the first block row emits
        # y_{t+1} = sum_k Phi_k y_{t+1-k}, the lower blocks shift history down.
        F = np.zeros((state_dim, state_dim))
        for lag in range(1, max_lag + 1):
            F[:n_channels, (lag - 1) * n_channels : lag * n_channels] = phi_lags[lag - 1]
        for block in range(1, max_lag):
            rows = slice(block * n_channels, (block + 1) * n_channels)
            F[rows, (block - 1) * n_channels : block * n_channels] = np.eye(n_channels)
        Q = np.zeros((state_dim, state_dim))
        Q[:n_channels, :n_channels] = np.diag(sigma**2)
        return F, Q

    phi_raw = np.zeros((max_lag, n_channels, n_channels))
    for target, source, lag, coef in couplings:
        phi_raw[lag - 1, target, source] = coef
    F_raw, Q_raw = build_companion(phi_raw, sigma_raw)
    p_raw = _stationary_cov(F_raw, Q_raw)
    # Var(y_i) is the state's own first block (the innovation is part of the state).
    scale = 1.0 / np.sqrt(np.diag(p_raw)[:n_channels])

    # Rescaled model: y_s = D y, so Phi_s = D Phi D^-1 and sigma_s = D sigma.
    phi_scaled = phi_raw * scale[None, :, None] / scale[None, None, :]
    sigma_scaled = scale * sigma_raw
    F, Q = build_companion(phi_scaled, sigma_scaled)
    if float(np.abs(np.linalg.eigvals(F)).max()) >= 1.0:
        raise RuntimeError("rescaled VAR companion is unstable; redesign the couplings")
    H = np.zeros((n_channels, state_dim))
    H[:, :n_channels] = np.eye(n_channels)
    R = np.zeros((n_channels, n_channels))
    P0 = _stationary_cov(F, Q)

    def simulate(rng: np.random.Generator, T: int) -> np.ndarray:
        state = rng.multivariate_normal(np.zeros(state_dim), P0)
        eps = rng.standard_normal((T, n_channels)) * sigma_scaled
        out = np.empty((T, n_channels))
        for t in range(T):
            out[t] = state[:n_channels]
            state = state @ F.T
            state[:n_channels] += eps[t]
        return out

    return SyntheticTask(
        name="lgssm_xchannel",
        train_diag=simulate(rng_main, sizes["train"] + sizes["diag"]),
        diag_start=sizes["train"],
        cert=simulate(rng_cert, _PREFIX + sizes["cert"]),
        test=simulate(rng_test, _PREFIX + sizes["test"]),
        oracle=KalmanOracle(F, Q, H, R, P0),
        structure={
            "n_channels": n_channels,
            "max_lag": max_lag,
            "couplings": [
                {"target": i, "source": j, "lag": lag, "coef": float(scale[i] * coef / scale[j])}
                for i, j, lag, coef in couplings
            ],
            "channel_innovation_sd": sigma_scaled.tolist(),
        },
    )


def hmm_ar_switch(seed: int, sizes: dict[str, int] | None = None) -> SyntheticTask:
    """C=1 Markov-switching AR(1): two sticky regimes, no observation noise.

    Regime s has mean mu_s in {0, 2} and AR coefficient phi_s in {0.3, 0.95};
    the transition matrix is sticky, so regimes persist for dozens of steps.  x_t
    is observed exactly, so the regime posterior follows the simple forward
    recursion and the h-step conditional mean is the joint recursion of
    _hmm_forecast (m_s = E[x 1{S=s}]) — exactly right, unlike mixing per-regime
    forecasts by posteriors.  The parameters are kept as stated (no rescaling):
    the stationary sd is ~1.2, already roughly unit.
    """
    sizes = _merge_sizes(sizes)
    rng_main, rng_cert, rng_test = _spawn(seed)

    trans = np.array([[0.99, 0.01], [0.02, 0.98]])
    phi = np.array([0.3, 0.95])
    mu = np.array([0.0, 2.0])
    sigma = 0.5
    pi = _markov_stationary(trans)

    def simulate(rng: np.random.Generator, T: int) -> np.ndarray:
        u = rng.random(T)
        # cumsum can land 1-2 ulps below 1.0; pinning the last entry keeps
        # searchsorted inside {0, ..., K-1} for every u in [0, 1)
        cdf = np.cumsum(trans, axis=1)
        cdf[:, -1] = 1.0
        cdf0 = np.cumsum(pi)
        cdf0[-1] = 1.0
        regimes = np.empty(T, dtype=np.int64)
        regimes[0] = int(np.searchsorted(cdf0, u[0]))
        for t in range(1, T):
            regimes[t] = int(np.searchsorted(cdf[regimes[t - 1]], u[t]))
        eps = rng.standard_normal(T)
        x = np.empty(T)
        prev = float(mu[regimes[0]]) + sigma / math.sqrt(1.0 - float(phi[regimes[0]]) ** 2) * eps[0]
        x[0] = prev
        for t in range(1, T):
            s = regimes[t]
            prev = float(mu[s]) + float(phi[s]) * (prev - float(mu[s])) + sigma * eps[t]
            x[t] = prev
        return x[:, None]

    return SyntheticTask(
        name="hmm_ar_switch",
        train_diag=simulate(rng_main, sizes["train"] + sizes["diag"]),
        diag_start=sizes["train"],
        cert=simulate(rng_cert, _PREFIX + sizes["cert"]),
        test=simulate(rng_test, _PREFIX + sizes["test"]),
        oracle=HmmOracle(trans, mu, phi, sigma, pi),
        structure={
            "n_regimes": 2,
            "means": mu.tolist(),
            "phis": phi.tolist(),
            "transition": trans.tolist(),
            "innovation_sd": sigma,
            "stationary_regime_prob": pi.tolist(),
        },
    )


def arfima(seed: int, sizes: dict[str, int] | None = None) -> SyntheticTask:
    """C=1 ARFIMA(0, d=0.35, 0) with unit innovation variance.

    Simulation is truncated fractional differencing with M = 2e6 MA terms
    (x_t = sum_{j<=M} psi_j eps_{t-j}); the discarded tail variance is
    M^{2d-1}/((1-2d) Gamma(d)^2) ~ 0.0066, i.e. 0.4% of gamma_0, and the ACF at
    the pipeline's longest lags stays within ~8% of the exact process.  Exact
    simulation (Davies-Harte circulant embedding) is NOT available at d = 0.35:
    the embedded covariance is indefinite (checked; negative eigenvalues of order
    1e-2 relative), which is the same regime where the method fails for fGn with
    H > 3/4.

    The oracle is DL(2000) on the theoretical autocovariance.  On the simulated
    truncated process its 1-step MSE measures ~1.003 against the process's own
    optimum ~1.0001, i.e. the simulation/oracle mismatch is negligible.  Origins
    with fewer than 2000 realized rows use the optimal AR of their available lag
    depth (within 0.02% of the full-depth MSE at the default burn-in, measured).
    Fresh segments carry a 2100-row prefix (>= the DL order), so the scored
    content rows always see the full lag set; only origins inside the burn-in
    window between 591 and 2000 use the truncated path.
    """
    sizes = _merge_sizes(sizes)
    rng_main, rng_cert, rng_test = _spawn(seed)

    def simulate(rng: np.random.Generator, T: int) -> np.ndarray:
        return _arfima_simulate(rng, T, _ARFIMA_D, _ARFIMA_MA_TRUNC)[:, None]

    oracle = ArfimaOracle()
    return SyntheticTask(
        name="arfima",
        train_diag=simulate(rng_main, sizes["train"] + sizes["diag"]),
        diag_start=sizes["train"],
        cert=simulate(rng_cert, _ARFIMA_PREFIX + sizes["cert"]),
        test=simulate(rng_test, _ARFIMA_PREFIX + sizes["test"]),
        oracle=oracle,
        structure={
            "d": _ARFIMA_D,
            "innovation_variance": 1.0,
            "dl_order": _ARFIMA_ORDER,
            "ma_truncation": _ARFIMA_MA_TRUNC,
            "gamma0": float(oracle.gamma[0]),
        },
    )


def garch(seed: int, sizes: dict[str, int] | None = None) -> SyntheticTask:
    """C=1 GARCH(1,1) with alpha=0.05, beta=0.9 and unit unconditional variance.

    A zero-mean martingale difference: the conditional mean is exactly zero at
    every horizon, so the oracle forecast is identically 0 — the mean-unpredictable
    negative control (the conditional *variance* is predictable, but that is not
    the target).  Starting the recursion at the stationary variance makes the
    simulation exactly stationary from the first row.
    """
    sizes = _merge_sizes(sizes)
    rng_main, rng_cert, rng_test = _spawn(seed)

    alpha, beta = 0.05, 0.9
    omega = 0.05  # (1 - alpha - beta) * 1: the target unconditional variance is 1

    def simulate(rng: np.random.Generator, T: int) -> np.ndarray:
        eps = rng.standard_normal(T)
        x = np.empty(T)
        h = 1.0  # omega / (1 - alpha - beta)
        prev = 0.0
        for t in range(T):
            h = omega + alpha * prev * prev + beta * h
            prev = math.sqrt(h) * eps[t]
            x[t] = prev
        return x[:, None]

    return SyntheticTask(
        name="garch",
        train_diag=simulate(rng_main, sizes["train"] + sizes["diag"]),
        diag_start=sizes["train"],
        cert=simulate(rng_cert, _PREFIX + sizes["cert"]),
        test=simulate(rng_test, _PREFIX + sizes["test"]),
        oracle=ZeroOracle(),
        structure={"alpha": alpha, "beta": beta, "omega": omega},
    )


def random_walk(seed: int, sizes: dict[str, int] | None = None) -> SyntheticTask:
    """C=1 random walk with increment sd 0.1; the oracle is the last observed value.

    Not variance-scaled: a random walk is non-stationary by design and the 0.1
    increment scale is already the stability-relevant quantity.
    """
    sizes = _merge_sizes(sizes)
    rng_main, rng_cert, rng_test = _spawn(seed)
    sigma = 0.1

    def simulate(rng: np.random.Generator, T: int) -> np.ndarray:
        return sigma * rng.standard_normal(T).cumsum()[:, None]

    return SyntheticTask(
        name="random_walk",
        train_diag=simulate(rng_main, sizes["train"] + sizes["diag"]),
        diag_start=sizes["train"],
        cert=simulate(rng_cert, _PREFIX + sizes["cert"]),
        test=simulate(rng_test, _PREFIX + sizes["test"]),
        oracle=LastValueOracle(sigma),
        structure={"sigma": sigma},
    )


# --- nonlinear cross-channel task -------------------------------------------------
#
# G0-6 screening showed lgssm_xchannel is linear-Gaussian: a linear corrector
# over realized history cashes its whole gap.  This task adds tanh couplings so
# part of the cross-channel dependence is only reachable by writing the
# nonlinear structure down.

_NLX_LAG_CHOICES = (3, 5, 7)
_NLX_CHANNELS = 4
_NLX_SOURCES_PER_CHANNEL = 2  # active cross pairs each channel receives (no self pairs)
_NLX_MAG_RANGE = (0.7, 1.3)  # relative per-pair magnitude spread; rescaling sets the level
# |b| / |a| per pair, drawn with a's sign (b_ij = ratio_ij * a_ij).  The contract
# asks the tanh terms to carry roughly half of the cross-channel dependence; at
# the unit channel variance this task is built to, tanh saturates
# (var(tanh(y)) = 0.394 var(y)) and the same-sign tanh term co-moves with the
# linear one (cov = 0.602 a b), so the variance share works out to
# 0.394 r^2 / (1 + 0.394 r^2 + 1.204 r).  This band centers it near 0.5; the
# naive 0.3-0.5 band would leave it near 5%.
_NLX_RATIO_RANGE = (2.8, 4.2)
# The one-step map linearized at the origin is a + b (tanh'(0) = 1); the joint
# rescale below puts its companion spectral radius exactly at this value.  Away
# from the origin tanh saturation only lowers the slopes, and the uniformly
# bounded tanh term keeps the process bounded whenever the linear part alone is
# stable, so this is the binding stability quantity.
_NLX_RADIUS_TARGET = 0.92
_NLX_LINEAR_RADIUS_CAP = 0.95  # contract-mandated cap on rho(companion(a))
# E[y tanh(y)] for y ~ N(0, 1): the slope of tanh's best linear approximation at
# unit scale (numerically integrated); linearizes the sigma-calibration proxy.
_NLX_TANH_SLOPE = 0.602
_NLX_BURN_IN = 2_000  # rows simulated and discarded in front of every stream
_NLX_PROBE_ROWS = 12_000  # sigma-calibration probe length (own burn-in included)
_NLX_PROBE_ROUNDS = 3
_NLX_MC_PATHS = 1_000  # default forward paths per origin for h > 1
_NLX_MC_PATH_CAP = 250_000  # path-origin cells per MC chunk; bounds memory only


def _nlx_coeff_by_lag(
    lags: np.ndarray, a: np.ndarray, b: np.ndarray
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """One (lag, linear, nonlinear) matrix group per distinct active lag.

    Inactive pairs carry lag 0 with zero coefficients, so they never reach the
    output; grouping by lag lets every consumer below touch each realized row
    exactly once per (linear, tanh) pair of terms.
    """
    return [
        (int(lag), a * (lags == lag), b * (lags == lag))
        for lag in np.unique(lags[lags > 0])
    ]


def _nlx_cross_mean(
    values_by_lag: list[np.ndarray],
    coeff: list[tuple[int, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Conditional-mean cross-channel sum_j a_ij y_j(t-lag_ij) + b_ij tanh(y_j(t-lag_ij)).

    ``values_by_lag[k]`` holds the [..., C] realized or simulated values at the
    k-th distinct lag of ``coeff``; the channel axis is last and any leading
    axes (single rows, origin batches, path batches) broadcast through the
    [C, C] matmuls, which is what keeps the simulator, the closed form and the
    Monte Carlo on one code path.
    """
    mean = None
    for values, (_, a_l, b_l) in zip(values_by_lag, coeff):
        term = values @ a_l.T + np.tanh(values) @ b_l.T
        mean = term if mean is None else mean + term
    return np.asarray(mean)


def _nlx_companion(coeff: list[tuple[int, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Companion matrix of the lag-block VAR; state is the last max_lag rows of y."""
    n_channels = coeff[0][1].shape[0]
    max_lag = max(lag for lag, _, _ in coeff)
    companion = np.zeros((n_channels * max_lag, n_channels * max_lag))
    for lag, a_l, _ in coeff:
        companion[:n_channels, (lag - 1) * n_channels : lag * n_channels] = a_l
    for block in range(1, max_lag):
        rows = slice(block * n_channels, (block + 1) * n_channels)
        companion[rows, (block - 1) * n_channels : block * n_channels] = np.eye(n_channels)
    return companion


def _nlx_rescale(
    lags: np.ndarray, a: np.ndarray, b: np.ndarray, target: float
) -> tuple[np.ndarray, np.ndarray]:
    """Common scale k for (a, b) so the companion of k*(a+b) has spectral radius == target.

    The radius is NOT linear in k (a lag-p term enters like k^(1/p)), so the
    scale is found by bisection; the dominant root grows monotonically in k.
    Scaling a and b together leaves every b_ij / a_ij ratio — and hence the
    designed nonlinear share — unchanged.
    """
    total = a + b
    coeff = _nlx_coeff_by_lag(lags, total, np.zeros_like(total))

    def radius_at(scale: float) -> float:
        scaled = [(lag, a_l * scale, b_l) for lag, a_l, b_l in coeff]
        return float(np.abs(np.linalg.eigvals(_nlx_companion(scaled))).max())

    lo, hi = 0.0, 1.0
    while radius_at(hi) < target:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if radius_at(mid) < target:
            lo = mid
        else:
            hi = mid
    scale = 0.5 * (lo + hi)
    return a * scale, b * scale


def _nlx_simulate(
    rng: np.random.Generator,
    coeff: list[tuple[int, np.ndarray, np.ndarray]],
    sigma: np.ndarray,
    rows: int,
) -> np.ndarray:
    """One stationary stream of ``rows`` content rows of the exact dynamics.

    A max_lag zero pre-history plus a _NLX_BURN_IN burn-in prefix is simulated
    and discarded in front; the burn-in is >= 2000 rows and the system contracts
    at rate _NLX_RADIUS_TARGET per step, so the transient is long dead by row 0.
    """
    n_channels = sigma.shape[0]
    max_lag = max(lag for lag, _, _ in coeff)
    total = _NLX_BURN_IN + rows
    y = np.zeros((total + max_lag, n_channels))
    eps = rng.standard_normal((total, n_channels)) * sigma
    lags = [lag for lag, _, _ in coeff]
    for t in range(max_lag, total + max_lag):
        y[t] = _nlx_cross_mean([y[t - lag] for lag in lags], coeff) + eps[t - max_lag]
    return y[max_lag + _NLX_BURN_IN :]


class NonlinearXChannelOracle:
    """Exact 1-step / Monte-Carlo h-step conditional mean of the tanh-coupled VAR.

    h = 1 is closed form (tanh of realized values).  For h > 1 the tanh composes
    across steps and no closed form exists, so every origin gets ``n_paths``
    forward simulations of the exact dynamics, vectorized over paths, and the
    forecast is the path mean.  The MC standard error of one forecast entry is
    sd(y_{o+h}) / sqrt(n_paths) ~ 0.03 at the default 1000 paths (conditional
    sd ~ 0.7-1.0, mean-reverting), so the oracle's own MSE carries an inflation
    of about sd^2 / n_paths ~ 1e-3 on a unit-variance channel — the tiny
    residual 'gap' a stage-0 check will attribute to the oracle; noted here per
    the contract.  The MC generator is rebuilt from the immutable ``mc_seq`` on
    every forecast call, so forecasts are bitwise reproducible under the same
    seed and idempotent under repeated calls.
    """

    def __init__(
        self,
        lags: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        sigma: np.ndarray,
        n_paths: int = _NLX_MC_PATHS,
        mc_seq: np.random.SeedSequence | None = None,
    ) -> None:
        self.lags = np.asarray(lags, dtype=np.int64)
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.sigma = np.asarray(sigma, dtype=np.float64)
        if not (self.lags.shape == self.a.shape == self.b.shape and self.lags.ndim == 2):
            raise ValueError(
                f"lags/a/b must share one [C, C] shape, got {self.lags.shape}, "
                f"{self.a.shape}, {self.b.shape}")
        if self.sigma.shape != (self.a.shape[0],):
            raise ValueError(
                f"sigma must be [{self.a.shape[0]}], got shape {self.sigma.shape}")
        if (self.lags < 0).any() or (self.lags > 0).sum() == 0:
            raise ValueError("lags must be zero for inactive pairs and positive elsewhere")
        # A coefficient on a lag-0 pair would read the target's own time row.
        if ((self.lags == 0) & ((self.a != 0) | (self.b != 0))).any():
            raise ValueError("inactive (lag-0) pairs must carry zero coefficients")
        if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)) or n_paths <= 0:
            raise ValueError(f"n_paths must be a positive int, got {n_paths!r}")
        self.n_paths = int(n_paths)
        self.max_lag = int(self.lags.max())
        self._coeff = _nlx_coeff_by_lag(self.lags, self.a, self.b)
        self.mc_seq = mc_seq if mc_seq is not None else np.random.SeedSequence(0)

    def one_step_mean(self, seg: np.ndarray, times: np.ndarray) -> np.ndarray:
        """Exact E[y_t | seg[0..t-1]] at each t in ``times``; returns [T, C]."""
        seg = np.asarray(seg, dtype=np.float64)
        times = np.asarray(times, dtype=np.int64)
        if seg.ndim != 2 or seg.shape[1] != self.a.shape[0]:
            raise ValueError(f"seg must be [T, {self.a.shape[0]}], got shape {seg.shape}")
        if times.ndim != 1:
            raise ValueError(f"times must be 1-D, got shape {times.shape}")
        out = np.empty((times.size, seg.shape[1]))
        if times.size == 0:
            return out
        if times.min() < self.max_lag:
            raise ValueError(f"time {times.min()} lacks the max_lag={self.max_lag} realized rows")
        if times.max() > seg.shape[0] - 1:
            raise ValueError(f"time {times.max()} lies beyond the end of seg")
        values = [seg[times - lag] for lag, _, _ in self._coeff]
        return _nlx_cross_mean(values, self._coeff)

    def path_means(
        self,
        seg: np.ndarray,
        origins: np.ndarray,
        horizon: int,
        n_paths: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Raw MC path means of the simulated steps 1..horizon; [B, horizon, C].

        ``forecast`` uses this for h >= 2 and overwrites h = 1 with the closed
        form; the method is public so tests can build independent brute-force
        estimates with their own rng and path count.
        """
        seg = np.asarray(seg, dtype=np.float64)
        origins = np.asarray(origins, dtype=np.int64)
        if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError(f"horizon must be a positive int, got {horizon!r}")
        if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)) or n_paths <= 0:
            raise ValueError(f"n_paths must be a positive int, got {n_paths!r}")
        if seg.ndim != 2 or seg.shape[1] != self.a.shape[0]:
            raise ValueError(f"seg must be [T, {self.a.shape[0]}], got shape {seg.shape}")
        if origins.ndim != 1:
            raise ValueError(f"origins must be 1-D, got shape {origins.shape}")
        out = np.empty((origins.size, int(horizon), seg.shape[1]))
        if origins.size == 0:
            return out
        if origins.min() < self.max_lag - 1:
            raise ValueError(
                f"origin {origins.min()} lacks the max_lag={self.max_lag} realized rows")
        if origins.max() + int(horizon) > seg.shape[0] - 1:
            raise ValueError(f"origin {origins.max()} has targets beyond the end of seg")
        # Chunk so each block's path tensor stays ~250k cells; accuracy is
        # unaffected (every origin still gets its own independent draws).
        chunk = max(1, _NLX_MC_PATH_CAP // int(n_paths))
        for start in range(0, origins.size, chunk):
            sel = origins[start : start + chunk]
            history = seg[sel[:, None] - np.arange(self.max_lag)]  # [B, max_lag, C], newest first
            # One deque entry per lag row, newest first: entry lag-1 is
            # y[o + step - lag] whatever the step, because maxlen drops the
            # oldest realized row exactly when the simulated window covers it.
            paths = collections.deque(
                [np.broadcast_to(history[:, k][:, None, :], (sel.size, n_paths, seg.shape[1]))
                 for k in range(self.max_lag)],
                maxlen=self.max_lag,
            )
            for step in range(1, int(horizon) + 1):
                values = [paths[lag - 1] for lag, _, _ in self._coeff]
                draws = _nlx_cross_mean(values, self._coeff) + rng.standard_normal(
                    (sel.size, n_paths, seg.shape[1])) * self.sigma
                out[start : start + sel.size, step - 1, :] = draws.mean(axis=1)
                paths.appendleft(draws)
        return out

    def forecast(self, seg: np.ndarray, origins: np.ndarray, cfg: HeadroomConfig) -> np.ndarray:
        seg, origins = _check_forecast_args(seg, origins, cfg)
        if cfg.horizon < 1:
            raise ValueError(f"horizon must be positive, got {cfg.horizon}")
        if seg.shape[1] != self.a.shape[0]:
            raise ValueError(f"seg has {seg.shape[1]} channels, model has {self.a.shape[0]}")
        out = np.empty((origins.size, cfg.horizon, seg.shape[1]))
        if origins.size == 0:
            return out
        if origins.min() < self.max_lag - 1:
            raise ValueError(
                f"origin {origins.min()} lacks the max_lag={self.max_lag} realized rows")
        # h = 1 exactly; the MC below also produces a step-1 mean, but the
        # closed form carries no sampling noise, so it wins.
        out[:, 0, :] = self.one_step_mean(seg, origins + 1)
        if cfg.horizon > 1:
            means = self.path_means(
                seg, origins, cfg.horizon, self.n_paths, np.random.default_rng(self.mc_seq))
            out[:, 1:, :] = means[:, 1:, :]
        return out

    def mse_theoretical(self) -> np.ndarray | None:
        # The h-step conditional variance composes tanh across steps (E[tanh] of
        # a Gaussian mixture is not a mixture of E[tanh]), so no closed form
        # exists; per the contract we return None rather than a surrogate.
        return None


def nonlinear_xchannel(seed: int, sizes: dict[str, int] | None = None) -> SyntheticTask:
    """C=4 nonlinear cross-channel VAR: tanh couplings at per-pair lags 3/5/7 (contract §8).

    Dynamics: y_{i,t} = sum_j a_ij y_{j,t-lag_ij} + sum_j b_ij tanh(y_{j,t-lag_ij})
    + eps_{i,t} with independent N(0, sigma_i^2) innovations.  Each channel
    receives _NLX_SOURCES_PER_CHANNEL randomly drawn cross pairs (no self
    pairs); every pair has its own lag from {3, 5, 7}, its own sign and
    magnitude, and its own b/a ratio drawn from _NLX_RATIO_RANGE with the same
    sign as a_ij (b_ij = ratio_ij * a_ij), which centers the tanh share of the
    1-step conditional mean's cross-channel contribution near one half.

    Stability: the tanh term is uniformly bounded, so rho(companion(a)) < 1
    makes the nonlinear process bounded no matter what b is; the construction
    instead rescales (a, b) jointly so the origin-linearized system
    companion(a + b) (tanh'(0) = 1) has spectral radius exactly
    _NLX_RADIUS_TARGET — the binding quantity for a centred stationary
    distribution — and additionally checks the contract's linear-part cap
    rho(companion(a)) <= _NLX_LINEAR_RADIUS_CAP (RuntimeError beyond it; the
    same-sign construction keeps rho(a) below rho(a+b) = target < cap).

    Channel scale: per-channel innovation sd is calibrated to roughly unit
    channel variance.  A linear proxy (tanh ~ _NLX_TANH_SLOPE * y at unit scale)
    gives a closed-form diagonal solve, and _NLX_PROBE_ROUNDS probe simulations
    (re-seeded every round, so sd(sigma) is a smooth deterministic function)
    correct for the curvature the proxy ignores; measured content sd lands
    within a few percent of 1 across seeds.

    Oracle: h=1 closed form, h>1 per-origin Monte Carlo through the exact
    dynamics (NonlinearXChannelOracle; MC standard error documented there).
    mse_theoretical() is None — no closed form exists.  Every stream is
    simulated with a _NLX_BURN_IN burn-in prefix discarded in front, then the
    usual train_diag / cert / test layout (cert/test fresh spawns with the
    standard _PREFIX), bitwise reproducible under the same seed.
    """
    sizes = _merge_sizes(sizes)
    rng_main, rng_cert, rng_test = _spawn(seed)
    # Children 0-2 are the same simulation streams _spawn hands out; 3-5 are
    # this task's own design / calibration / MC streams.
    children = np.random.SeedSequence(int(seed)).spawn(6)
    design = np.random.default_rng(children[3])
    probe_seq = children[4]
    mc_seq = children[5]

    lags = np.zeros((_NLX_CHANNELS, _NLX_CHANNELS), dtype=np.int64)
    a = np.zeros((_NLX_CHANNELS, _NLX_CHANNELS))
    b = np.zeros((_NLX_CHANNELS, _NLX_CHANNELS))
    for target_channel in range(_NLX_CHANNELS):
        sources = design.choice(
            [j for j in range(_NLX_CHANNELS) if j != target_channel],
            size=_NLX_SOURCES_PER_CHANNEL, replace=False)
        for source in sources:
            lags[target_channel, source] = design.choice(_NLX_LAG_CHOICES)
            magnitude = design.uniform(*_NLX_MAG_RANGE)
            sign = design.choice((-1.0, 1.0))
            a[target_channel, source] = magnitude * sign
            b[target_channel, source] = (
                design.uniform(*_NLX_RATIO_RANGE) * magnitude * sign)

    a, b = _nlx_rescale(lags, a, b, _NLX_RADIUS_TARGET)
    coeff = _nlx_coeff_by_lag(lags, a, b)
    linear_radius = float(np.abs(np.linalg.eigvals(_nlx_companion(coeff))).max())
    if linear_radius > _NLX_LINEAR_RADIUS_CAP:
        raise RuntimeError(
            f"linear part spectral radius {linear_radius:.3f} exceeds the "
            f"{_NLX_LINEAR_RADIUS_CAP} cap; redesign the coefficient draw")

    # Innovation sd, proxy first: with tanh linearized at unit scale the system
    # is a plain VAR whose stationary covariance scales exactly with the
    # innovation sd, so one Lyapunov solve gives unit marginals in the proxy.
    proxy = _nlx_coeff_by_lag(lags, a + _NLX_TANH_SLOPE * b, np.zeros_like(b))
    proxy_companion = _nlx_companion(proxy)
    noise = np.zeros_like(proxy_companion)
    noise[:_NLX_CHANNELS, :_NLX_CHANNELS] = np.eye(_NLX_CHANNELS)
    sigma = 1.0 / np.sqrt(np.diag(_stationary_cov(proxy_companion, noise))[:_NLX_CHANNELS])
    for _ in range(_NLX_PROBE_ROUNDS):
        probe = _nlx_simulate(np.random.default_rng(probe_seq), coeff, sigma, _NLX_PROBE_ROWS)
        sigma = sigma / probe.std(axis=0)
    if not np.isfinite(sigma).all() or (sigma <= 0).any():
        raise RuntimeError(
            f"innovation-sd calibration diverged (sigma={sigma.tolist()}); "
            "the drawn couplings are unstable")

    def simulate(rng: np.random.Generator, rows: int) -> np.ndarray:
        return _nlx_simulate(rng, coeff, sigma, rows)

    train_diag = simulate(rng_main, sizes["train"] + sizes["diag"])
    oracle = NonlinearXChannelOracle(lags, a, b, sigma, mc_seq=mc_seq)

    # Measured share of the 1-step conditional mean's cross-channel part carried
    # by the tanh terms, pooled over channels and time on train_diag.
    times = np.arange(int(lags.max()), train_diag.shape[0])
    values = [train_diag[times - lag] for lag, _, _ in coeff]
    linear_part = _nlx_cross_mean(
        values, [(lag, a_l, np.zeros_like(b_l)) for lag, a_l, b_l in coeff])
    nonlinear_part = _nlx_cross_mean(
        values, [(lag, np.zeros_like(a_l), b_l) for lag, a_l, b_l in coeff])
    nonlinear_fraction = float(nonlinear_part.var() / (linear_part + nonlinear_part).var())

    active = list(zip(*np.nonzero(lags)))
    return SyntheticTask(
        name="nonlinear_xchannel",
        train_diag=train_diag,
        diag_start=sizes["train"],
        cert=simulate(rng_cert, _PREFIX + sizes["cert"]),
        test=simulate(rng_test, _PREFIX + sizes["test"]),
        oracle=oracle,
        structure={
            "n_channels": _NLX_CHANNELS,
            "max_lag": int(lags.max()),
            "cross_lags": {
                f"{int(i)}->{int(j)}": int(lags[i, j]) for i, j in active
            },
            "couplings": [
                {"target": int(i), "source": int(j), "lag": int(lags[i, j]),
                 "linear": float(a[i, j]), "nonlinear": float(b[i, j])}
                for i, j in active
            ],
            "nonlinear_fraction": nonlinear_fraction,
            "channel_innovation_sd": sigma.tolist(),
        },
    )


REGISTRY: dict[str, Callable[..., SyntheticTask]] = {
    "lgssm_multiperiod": lgssm_multiperiod,
    "lgssm_xchannel": lgssm_xchannel,
    "hmm_ar_switch": hmm_ar_switch,
    "arfima": arfima,
    "garch": garch,
    "random_walk": random_walk,
    "nonlinear_xchannel": nonlinear_xchannel,
}
