from __future__ import annotations

import functools
import itertools
import math

import numpy as np
import pytest

from harnessing_ts.headroom.config import HeadroomConfig
from harnessing_ts.headroom.synthetic import (
    _ARFIMA_ORDER,
    _ARFIMA_PREFIX,
    _PREFIX,
    DEFAULT_SIZES,
    REGISTRY,
    NonlinearXChannelOracle,
    SyntheticTask,
    _arfima_gamma,
    _hmm_forecast,
)

SMALL = {"train": 400, "diag": 300, "cert": 500, "test": 400}
CHANNELS = {
    "lgssm_multiperiod": 4,
    "lgssm_xchannel": 4,
    "hmm_ar_switch": 1,
    "arfima": 1,
    "garch": 1,
    "random_walk": 1,
    "nonlinear_xchannel": 4,
}
CFG = HeadroomConfig()  # horizon 24, burn_in 591
CFG_TINY = HeadroomConfig(input_length=8, horizon=6)
CFG_H4 = HeadroomConfig(input_length=8, horizon=4)


@functools.lru_cache(maxsize=None)
def build(name: str, sizes: tuple[tuple[str, int], ...], seed: int) -> SyntheticTask:
    """Shared builds keep the file fast; tests never mutate tasks."""
    return REGISTRY[name](seed, dict(sizes))


def small_task(name: str, seed: int = 0) -> SyntheticTask:
    return build(name, tuple(sorted(SMALL.items())), seed)


def usable_origins(seg: np.ndarray, cfg: HeadroomConfig = CFG) -> np.ndarray:
    first, last = cfg.burn_in, seg.shape[0] - cfg.horizon - 1
    return np.arange(first, last + 1)


def test_registry_matches_contract_and_sizes_validate() -> None:
    assert set(REGISTRY) == set(CHANNELS)
    assert DEFAULT_SIZES == {"train": 20_000, "diag": 10_000, "cert": 12_000, "test": 20_000}
    task = REGISTRY["random_walk"](0, {"cert": 64})
    assert task.cert.shape[0] == _PREFIX + 64
    with pytest.raises(ValueError, match="unknown size key"):
        REGISTRY["garch"](0, {"train": 10, "nope": 5})
    with pytest.raises(ValueError, match="positive int"):
        REGISTRY["garch"](0, {"train": 0})
    with pytest.raises(ValueError, match="seed"):
        REGISTRY["garch"](-1, SMALL)


@pytest.mark.parametrize("name", sorted(CHANNELS))
def test_same_seed_is_bitwise_reproducible(name: str) -> None:
    first = small_task(name, seed=0)
    second = REGISTRY[name](0, SMALL)
    np.testing.assert_array_equal(first.train_diag, second.train_diag)
    np.testing.assert_array_equal(first.cert, second.cert)
    np.testing.assert_array_equal(first.test, second.test)
    assert first.diag_start == second.diag_start


def test_different_seeds_produce_different_draws() -> None:
    a = small_task("random_walk", seed=0).test
    b = small_task("random_walk", seed=1).test
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("name", sorted(CHANNELS))
def test_shapes_channels_and_forecast_output(name: str) -> None:
    task = small_task(name)
    c = CHANNELS[name]
    prefix = _ARFIMA_PREFIX if name == "arfima" else _PREFIX
    if name == "arfima":
        # fresh segments must give the DL oracle its full lag set from the
        # first usable origin onwards (and the usual burn-in prefix besides)
        assert _ARFIMA_PREFIX >= _ARFIMA_ORDER
        assert _ARFIMA_PREFIX >= CFG.burn_in + CFG.input_length
    assert task.train_diag.shape == (SMALL["train"] + SMALL["diag"], c)
    assert task.diag_start == SMALL["train"]
    assert task.cert.shape == (prefix + SMALL["cert"], c)
    assert task.test.shape == (prefix + SMALL["test"], c)
    for arr in (task.train_diag, task.cert, task.test):
        assert arr.dtype == np.float64
        assert np.isfinite(arr).all()

    origins = np.array([prefix, prefix + 17, task.cert.shape[0] - CFG.horizon - 1])
    pred = task.oracle.forecast(task.cert, origins, CFG)
    assert pred.shape == (3, CFG.horizon, c)
    assert np.isfinite(pred).all()
    empty = task.oracle.forecast(task.cert, np.array([], dtype=np.int64), CFG)
    assert empty.shape == (0, CFG.horizon, c)

    mse = task.oracle.mse_theoretical()
    if name in {"lgssm_multiperiod", "lgssm_xchannel", "random_walk"}:
        assert mse is not None
        assert mse.shape == (HeadroomConfig().horizon,)
        assert (mse > 0).all()
        assert (np.diff(mse) >= -1e-12).all()  # h-step MSE is nondecreasing in h
    else:
        assert mse is None


def _exact_gaussian_forecast(
    task: SyntheticTask, t_rows: int, origin: int, horizon: int
) -> np.ndarray:
    """Conditional mean of y[o+1..o+H] given y[0..o] from the exact joint covariance.

    The state path z_t = A z_{t-1} + w_t with z_0 ~ N(0, P0) gives
    Cov(z_t, z_s) = A^{t-s} Var(z_s) for t >= s, so the whole observation
    covariance follows from the marginal variances Var(z_s) and powers of A.
    """
    oracle = task.oracle
    A, Q, H, R, P0 = oracle.A, oracle.Q, oracle.H, oracle.R, oracle.P0
    n, c = A.shape[0], H.shape[0]
    marginals = [np.array(P0, dtype=np.float64)]
    for _ in range(t_rows - 1):
        marginals.append(A @ marginals[-1] @ A.T + Q)
    powers = [np.eye(n)]
    for _ in range(t_rows - 1):
        powers.append(powers[-1] @ A)
    blocks = np.empty((t_rows, t_rows, n, n))
    for t in range(t_rows):
        for s in range(t_rows):
            if t >= s:
                blocks[t, s] = powers[t - s] @ marginals[s]
            else:
                blocks[t, s] = (powers[s - t] @ marginals[t]).T
    cov_y = np.einsum("ij,tsjk,kl->tsil", H, blocks, H.T)
    cov_y[np.arange(t_rows), np.arange(t_rows)] += R
    flat = cov_y.transpose(0, 2, 1, 3).reshape(t_rows * c, t_rows * c)
    y = task.cert[:t_rows].reshape(-1)
    past = [t * c + ch for t in range(origin + 1) for ch in range(c)]
    future = [t * c + ch for t in range(origin + 1, origin + 1 + horizon) for ch in range(c)]
    solved = np.linalg.solve(flat[np.ix_(past, past)], y[past])
    return flat[np.ix_(future, past)] @ solved


@pytest.mark.parametrize(
    ("name", "t_rows", "origin"),
    [("lgssm_multiperiod", 30, 18), ("lgssm_xchannel", 24, 12)],
)
def test_lgssm_kalman_matches_exact_gaussian_conditioning(name: str, t_rows: int, origin: int) -> None:
    task = small_task(name)
    expected = _exact_gaussian_forecast(task, t_rows, origin, CFG_TINY.horizon)
    got = task.oracle.forecast(task.cert[:t_rows], np.array([origin]), CFG_TINY)[0]
    np.testing.assert_allclose(got, expected.reshape(CFG_TINY.horizon, -1), atol=1e-8)


def test_xchannel_oracle_matches_direct_var_recursion() -> None:
    task = small_task("lgssm_xchannel")
    c = CHANNELS["lgssm_xchannel"]
    phi_lags = np.zeros((task.structure["max_lag"], c, c))
    for entry in task.structure["couplings"]:
        phi_lags[entry["lag"] - 1, entry["target"], entry["source"]] = entry["coef"]
    seg = task.cert
    for origin in (_PREFIX + 8, _PREFIX + 90):
        means = {t: seg[t].copy() for t in range(origin + 1)}
        for h in range(1, CFG_TINY.horizon + 1):
            means[origin + h] = sum(phi_lags[k] @ means[origin + h - 1 - k] for k in range(len(phi_lags)))
        direct = np.stack([means[origin + h] for h in range(1, CFG_TINY.horizon + 1)])
        got = task.oracle.forecast(seg, np.array([origin]), CFG_TINY)[0]
        np.testing.assert_allclose(got, direct, atol=1e-8)
    sigma = np.array(task.structure["channel_innovation_sd"])
    # R = 0 and a fully observed state mean the 1-step error is the innovation itself
    np.testing.assert_allclose(task.oracle.mse_theoretical()[0], float((sigma**2).mean()), rtol=1e-8)


def _hmm_brute_force(
    p: np.ndarray, m: np.ndarray, trans: np.ndarray, mu: np.ndarray,
    phi: np.ndarray, horizon: int,
) -> np.ndarray:
    """Enumerate every regime path from the joint (p, m) at the origin."""
    k = trans.shape[0]
    implied_x = np.divide(m, p, out=np.zeros(k), where=p > 0)
    out = np.empty(horizon)
    for h in range(1, horizon + 1):
        total = 0.0
        for path in itertools.product(range(k), repeat=h):
            for s0 in range(k):
                prob = p[s0]
                prev = s0
                for s in path:
                    prob *= trans[prev, s]
                    prev = s
                value = implied_x[s0]
                for s in path:
                    value = mu[s] + phi[s] * (value - mu[s])
                total += prob * value
        out[h - 1] = total
    return out


def test_hmm_joint_recursion_matches_brute_force() -> None:
    trans = np.array([[0.93, 0.07], [0.05, 0.95]])
    mu = np.array([0.3, 1.7])
    phi = np.array([0.4, 0.9])
    p = np.array([0.3, 0.7])
    m = np.array([0.1, 1.4])  # arbitrary joint moment, not just p * x
    exact = _hmm_forecast(p[None, :], m[None, :], trans, mu, phi, 4)[0]
    brute = _hmm_brute_force(p, m, trans, mu, phi, 4)
    np.testing.assert_allclose(exact, brute, atol=1e-9)


def _hmm_posterior_by_enumeration(
    x: np.ndarray, trans: np.ndarray, mu: np.ndarray, phi: np.ndarray,
    sigma: float, pi: np.ndarray, t: int,
) -> np.ndarray:
    """P(S_t | x[0..t]) by summing over all 2^(t+1) regime paths.

    Uses the same convention as the oracle's filter: the chain starts from its
    stationary distribution and the first scored emission is x[1] | x[0].
    """
    k = trans.shape[0]
    values = np.zeros(k)
    for path in itertools.product(range(k), repeat=t + 1):
        logp = math.log(pi[path[0]])
        for step in range(1, t + 1):
            prev, s = path[step - 1], path[step]
            logp += math.log(trans[prev, s])
            mean = mu[s] + phi[s] * (x[step - 1] - mu[s])
            logp += -0.5 * (x[step] - mean) ** 2 / sigma**2
        values[path[-1]] += math.exp(logp)
    return values / values.sum()


def test_hmm_oracle_matches_brute_force_end_to_end() -> None:
    task = small_task("hmm_ar_switch")
    oracle = task.oracle
    x = task.cert[:, 0]
    for origin in (8, 12):  # 2^9 and 2^13 regime paths: small enough to enumerate
        posterior = _hmm_posterior_by_enumeration(
            x, oracle.trans, oracle.mu, oracle.phi, oracle.sigma, oracle.pi, origin)
        brute = _hmm_brute_force(
            posterior, posterior * x[origin], oracle.trans, oracle.mu, oracle.phi, CFG_H4.horizon)
        got = task.oracle.forecast(task.cert, np.array([origin]), CFG_H4)[0, :, 0]
        np.testing.assert_allclose(got, brute, atol=1e-9)


def test_lgssm_innovations_uncorrelated_and_mse_matches_theory() -> None:
    sizes = tuple(sorted({"train": 50, "diag": 50, "cert": 4_000, "test": 50}.items()))
    task = build("lgssm_multiperiod", sizes, 2)
    seg = task.cert
    origins = usable_origins(seg)
    pred = task.oracle.forecast(seg, origins, CFG)[:, 0, :]
    resid = seg[origins + 1] - pred
    # corr se ~ 1/sqrt(B) = 0.016 at B ~ 3400; 0.1 is a >6-sigma bound over the
    # 32 (lag, channel) pairs checked, and rejects any systematic mis-prediction
    for k in range(1, 9):
        for channel in range(seg.shape[1]):
            corr = np.corrcoef(resid[:, channel], seg[origins + 1 - k, channel])[0, 1]
            assert abs(corr) < 0.1
    # steady-state reference vs realized 1-step MSE on usable origins; the filter
    # transient at burn_in costs ~0.2% (measured) and sampling noise ~1.3%
    empirical = float((resid**2).mean())
    assert abs(empirical / task.oracle.mse_theoretical()[0] - 1.0) < 0.05


def test_arfima_dl_recovers_innovation_variance_and_residuals_are_clean() -> None:
    sizes = tuple(sorted({"train": 50, "diag": 50, "cert": 6_000, "test": 50}.items()))
    task = build("arfima", sizes, 1)
    oracle = task.oracle
    gamma = _arfima_gamma(0.35, _ARFIMA_ORDER)
    np.testing.assert_array_equal(oracle.gamma, gamma)
    assert oracle.nus[1] == pytest.approx(gamma[0] - gamma[1] ** 2 / gamma[0], rel=1e-12)
    # the gamma_0 normalization constant is correct iff DL(2000) on gamma returns
    # the process's unit innovation variance (an unnormalized sequence gives 0.46)
    assert oracle.nus[_ARFIMA_ORDER] == pytest.approx(1.0, abs=1e-2)

    seg = task.cert
    origins = usable_origins(seg)
    pred = oracle.forecast(seg, origins, CFG)[:, 0, 0]
    resid = seg[origins + 1, 0] - pred
    # corr se ~ 1/sqrt(B) = 0.013 at B ~ 5900 (origins below the DL depth use the
    # optimal shorter lag set, which is why this covers every usable origin)
    for k in range(1, 9):
        corr = np.corrcoef(resid, seg[origins + 1 - k, 0])[0, 1]
        assert abs(corr) < 0.08
    # sample variance against the theoretical gamma_0; the 2e6-term simulation
    # truncation costs 0.4% and long-memory sampling noise is ~7% at 6k rows
    content = seg[_ARFIMA_PREFIX:]
    assert abs(float(content.var()) / gamma[0] - 1.0) < 0.2


def _test_local_durbin_levinson(gamma: np.ndarray, order: int) -> list[np.ndarray]:
    """Independent DL used only to verify the oracle's short-history path."""
    phis: list[np.ndarray] = []
    phi = np.array([gamma[1] / gamma[0]])
    phis.append(phi)
    nu = gamma[0] * (1.0 - phi[0] ** 2)
    for m in range(2, order + 1):
        r = gamma[m] - phi @ gamma[m - 1 : 0 : -1]
        km = r / nu
        nu = nu * (1.0 - km * km)
        new = np.empty(m)
        new[: m - 1] = phi - km * phi[::-1]
        new[m - 1] = km
        phi = new
        phis.append(phi)
    return phis


def test_arfima_short_history_truncates_the_lag_set_gracefully() -> None:
    task = small_task("arfima")
    order = 40
    phis = _test_local_durbin_levinson(_arfima_gamma(0.35, order), order)
    seg = task.cert
    origins = np.array([5, 20, 39])
    got = task.oracle.forecast(seg[:200], origins, CFG_TINY)
    for i, origin in enumerate(origins):
        phi = phis[origin]  # origin o has o+1 realized rows -> AR(o+1)
        buffer = [float(seg[origin - j, 0]) for j in range(origin + 1)]
        expected = []
        for _ in range(CFG_TINY.horizon):
            value = float(np.dot(phi, buffer))
            expected.append(value)
            buffer.insert(0, value)
            buffer.pop()  # the recursion only ever uses the newest len(phi) values
        np.testing.assert_allclose(got[i, :, 0], expected, atol=1e-10)
    # adding future rows must not move forecasts at earlier origins
    longer = task.oracle.forecast(seg[:600], origins, CFG_TINY)
    np.testing.assert_array_equal(got, longer)


def test_negative_control_oracles() -> None:
    garch = small_task("garch")
    origins = np.array([_PREFIX, _PREFIX + 10])
    pred = garch.oracle.forecast(garch.cert, origins, CFG)
    assert pred.shape == (2, CFG.horizon, 1)
    assert (pred == 0.0).all()
    assert garch.oracle.mse_theoretical() is None

    walk = small_task("random_walk")
    pred = walk.oracle.forecast(walk.cert, origins, CFG)
    assert (pred[:, :, 0] == walk.cert[origins, 0][:, None]).all()
    sigma = walk.structure["sigma"]
    np.testing.assert_allclose(
        walk.oracle.mse_theoretical(), sigma**2 * np.arange(1, CFG.horizon + 1))


def test_forecast_rejects_bad_arguments() -> None:
    task = small_task("garch")
    with pytest.raises(ValueError, match="seg must be"):
        task.oracle.forecast(np.zeros(5), np.array([1]), CFG)
    with pytest.raises(ValueError, match="before the start"):
        task.oracle.forecast(task.cert, np.array([-1]), CFG)
    with pytest.raises(ValueError, match="beyond the end"):
        task.oracle.forecast(task.cert, np.array([task.cert.shape[0] - 1]), CFG)
    multiperiod = small_task("lgssm_multiperiod")
    with pytest.raises(ValueError, match="channels"):
        multiperiod.oracle.forecast(multiperiod.cert[:, :2], np.array([_PREFIX]), CFG)


# --- nonlinear_xchannel (tanh-coupled cross-channel VAR) -------------------------

NLX_MC_SIZES = tuple(sorted({"train": 50, "diag": 50, "cert": 2_500, "test": 50}.items()))
CFG_H8 = HeadroomConfig(input_length=8, horizon=8)
CFG_H6 = HeadroomConfig(input_length=8, horizon=6)


def nlx_mc_task(seed: int = 7) -> SyntheticTask:
    """Cert-heavy build shared by the exactness checks (h > 1 costs MC time)."""
    return build("nonlinear_xchannel", NLX_MC_SIZES, seed)


def test_nonlinear_xchannel_same_seed_reproducible_forecasts() -> None:
    first = small_task("nonlinear_xchannel")
    second = REGISTRY["nonlinear_xchannel"](0, SMALL)
    origins = np.array([_PREFIX + 5, _PREFIX + 61, _PREFIX + 130])
    left = first.oracle.forecast(first.cert, origins, CFG_H8)
    right = second.oracle.forecast(second.cert, origins, CFG_H8)
    np.testing.assert_array_equal(left, right)
    # the MC generator is rebuilt from the task's immutable mc_seq on every call,
    # so repeated forecasts on one task are bitwise identical as well
    np.testing.assert_array_equal(left, first.oracle.forecast(first.cert, origins, CFG_H8))
    # the h=1 slice is the closed form and carries no MC noise
    np.testing.assert_array_equal(left[:, 0, :], first.oracle.one_step_mean(
        first.cert, origins + 1))


def test_nonlinear_xchannel_one_step_oracle_is_exact() -> None:
    task = nlx_mc_task()
    oracle = task.oracle
    seg = task.cert
    origins = np.array([_PREFIX + 10, _PREFIX + 200, _PREFIX + 900])
    closed = oracle.one_step_mean(seg, origins + 1)
    mc = oracle.path_means(seg, origins, 1, 200_000, np.random.default_rng(11))
    # expected MS gap = sigma^2 / n_paths ~ 4e-6 (the MC noise floor); 1e-4
    # leaves an order of magnitude of headroom while still catching any
    # lag-indexing slip, which would err at O(0.1)
    assert float(((closed - mc[:, 0, :]) ** 2).mean()) < 1e-4

    times = np.arange(_PREFIX, _PREFIX + 1_600)
    resid = seg[times + 1] - oracle.one_step_mean(seg, times + 1)
    # exact 1-step residuals are the innovations themselves; corr se ~
    # 1/sqrt(1600) = 0.025, so 0.1 is ~4 sigma per pair (the max over the 32
    # checked pairs sits near 2.2 sigma)
    for k in range(1, 9):
        for channel in range(seg.shape[1]):
            corr = np.corrcoef(resid[:, channel], seg[times + 1 - k, channel])[0, 1]
            assert abs(corr) < 0.1


def test_nonlinear_xchannel_h_step_matches_independent_mc() -> None:
    task = nlx_mc_task()
    seg = task.cert
    origins = np.array([_PREFIX + 33, _PREFIX + 133, _PREFIX + 533, _PREFIX + 1_033])
    got = task.oracle.forecast(seg, origins, CFG_H6)
    # brute force with a different seed and 200x the oracle's 1000 paths
    brute = task.oracle.path_means(seg, origins, 6, 200_000, np.random.default_rng(4_242))
    # combined MC se ~ sqrt(sd^2/1000 + sd^2/200000) <= 0.033 (sd(y_{o+h}) <= 1.05,
    # mean-reverting); 0.2 is ~6 sigma
    assert float(np.abs(got[:, 1:, :] - brute[:, 1:, :]).max()) < 0.2
    # h=1: closed form vs brute MC, only the brute's own ~0.002 se remains
    assert float(np.abs(got[:, 0, :] - brute[:, 0, :]).max()) < 0.05


def test_nonlinear_xchannel_default_seed_stays_bounded() -> None:
    task = REGISTRY["nonlinear_xchannel"](0)  # default sizes, default seed
    for arr in (task.train_diag, task.cert, task.test):
        assert np.isfinite(arr).all()
        assert float(np.abs(arr).max()) < 50
    # the sigma calibration's contract: channels roughly unit variance
    np.testing.assert_allclose(task.cert[_PREFIX:].std(axis=0), 1.0, rtol=0.1)


def test_nonlinear_xchannel_tanh_carries_conditional_mean_variance() -> None:
    task = small_task("nonlinear_xchannel")
    oracle = task.oracle
    c = CHANNELS["nonlinear_xchannel"]
    lags = np.zeros((c, c), dtype=np.int64)
    a = np.zeros((c, c))
    b = np.zeros((c, c))
    for entry in task.structure["couplings"]:
        i, j = entry["target"], entry["source"]
        lags[i, j] = entry["lag"]
        a[i, j] = entry["linear"]
        b[i, j] = entry["nonlinear"]

    # decompose the 1-step conditional mean's cross-channel contribution on
    # train_diag into its linear and tanh parts, entry by entry
    y = task.train_diag
    t = np.arange(task.structure["max_lag"], y.shape[0])
    lin = np.zeros((t.size, c))
    non = np.zeros((t.size, c))
    for i, j in zip(*np.nonzero(lags)):
        lin[:, i] += a[i, j] * y[t - lags[i, j], j]
        non[:, i] += b[i, j] * np.tanh(y[t - lags[i, j], j])
    fraction = float(non.var() / (lin + non).var())
    assert fraction > 0.2  # the contract's floor; the design centers it near 0.5
    assert task.structure["nonlinear_fraction"] == pytest.approx(fraction, rel=1e-6)

    # a model with b zeroed misses exactly the tanh part, so its 1-step MSE
    # exceeds the oracle's by ~var(non) — the "measurable difference" between
    # explaining the linear part alone and linear+tanh
    linear_only = NonlinearXChannelOracle(lags, a, np.zeros_like(b), oracle.sigma)
    seg = task.cert
    times = np.arange(_PREFIX, seg.shape[0] - 1)
    realized = seg[times + 1]
    mse_full = float(((realized - oracle.one_step_mean(seg, times + 1)) ** 2).mean())
    mse_linear = float(((realized - linear_only.one_step_mean(seg, times + 1)) ** 2).mean())
    assert mse_linear - mse_full > 0.02  # ~var(non) ~ 0.076 (seed 0, measured)
    assert (mse_linear - mse_full) / mse_linear > 0.03
