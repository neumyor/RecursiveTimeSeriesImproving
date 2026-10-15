from __future__ import annotations

"""Synthetic tasks whose full-information optimal forecasters are exactly computable (contract §8)."""

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
    """C=4 LGSSM: a slow level, exact 24- and 168-period oscillators, colored noise.

    Latent state (5): an AR(1) level with phi = 0.995, plus a sin/cos rotation pair
    for each period.  The oscillators carry no process noise, so the periods stay
    exactly 24 and 168 forever (a rotation of a random initial draw — stationary
    because the isotropic prior is rotation-invariant).  Each channel observes a
    different linear mix of the latents plus its own AR(1) channel noise and white
    observation noise; channel 3 sees no level (partial observation).  All
    parameters are known, so a Kalman filter run from segment index 0 with the
    exact stationary prior followed by h-step mean propagation is the exact
    conditional mean.  Channels are scaled to unit variance through the known
    theoretical per-channel variance.
    """
    sizes = _merge_sizes(sizes)
    rng_main, rng_cert, rng_test = _spawn(seed)

    phi_level = 0.995
    sigma_level = 0.03
    periods = (24, 168)
    oscillator_var = 0.25  # prior variance of each sin/cos component
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
        a_latent[r, r] = math.cos(angle)
        a_latent[r, c] = math.sin(angle)
        a_latent[c, r] = -math.sin(angle)
        a_latent[c, c] = math.cos(angle)
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


REGISTRY: dict[str, Callable[..., SyntheticTask]] = {
    "lgssm_multiperiod": lgssm_multiperiod,
    "lgssm_xchannel": lgssm_xchannel,
    "hmm_ar_switch": hmm_ar_switch,
    "arfima": arfima,
    "garch": garch,
    "random_walk": random_walk,
}
