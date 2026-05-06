"""
processes_ql.py — QuantLib-backed stochastic price process classes
===================================================================
Three process classes for gas storage Monte Carlo valuation, all returning
a PathBundle so they are interchangeable inside GasStorageSimulator and the
forthcoming LSMCOptimiser.

    ┌─────────────────────────────────────────────────────────────────┐
    │  Process               QL backend                    state2     │
    ├─────────────────────────────────────────────────────────────────┤
    │  LognormalOUProcessQL  ExtendedOrnsteinUhlenbeck     None       │
    │  TwoFactorSchwartzQL   StochasticProcessArray        χ_t paths  │
    │                        (ExtOU × OrnsteinUhlenbeck)              │
    │  HestonProcessQL       HestonProcess                 v_t paths  │
    └─────────────────────────────────────────────────────────────────┘

PathBundle
----------
A lightweight dataclass carrying the simulation output in a common format.
`spots`  — always (n_paths, n_steps) EUR/MWh prices
`state2` — second factor, or None for single-factor models
`model`  — string identifier used by basis functions and diagnostics

Process interface
-----------------
Every process class exposes a single public method:

    simulate(date_grid: List[date]) -> PathBundle

The date_grid is the same daily List[date] produced by _build_date_grid()
in gas_storage_mc.py, so no separate grid construction is needed here.

RNG strategy
------------
All three classes share the same two-generator pattern from ou_quantlib.py:
  - Pseudo-random : GaussianPathGenerator / GaussianMultiPathGenerator
                    backed by MersenneTwister (use_sobol=False)
  - Quasi-random  : GaussianSobolPathGenerator / GaussianSobolMultiPathGenerator
                    backed by Sobol low-discrepancy sequences (use_sobol=True)

The number of Sobol dimensions equals factors × n_steps:
  - LognormalOU : 1 × n_steps
  - TwoFactor   : 2 × n_steps
  - Heston      : 2 × n_steps

Calibration notes
-----------------
LognormalOU
  kappa, sigma calibrated from market: AR(1) regression on daily log-returns
  for kappa; implied vol surface or historical vol for sigma.

TwoFactorSchwartz
  Parameters (kappa_xi, sigma_xi, kappa_eta, sigma_eta, rho, lambda_xi, lambda_eta)
  follow Schwartz-Smith (2000).  The initial state (xi0, eta0) is recovered
  from the current spot and a reference long-term forward.
  Forward curve repricing is approximate (flat long-term state); a full
  Schwartz-Smith calibration routine should fit (xi0, eta0, lambda_xi,
  lambda_eta) to the observed forward curve.

HestonProcess
  v0 is the current instantaneous variance.  (kappa, theta, sigma, rho) are
  calibrated jointly from the implied vol surface via QuantLib's HestonModel
  calibration helpers (not included here — supply pre-calibrated values).

Dependencies
------------
    pip install QuantLib numpy
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

import numpy as np
import QuantLib as ql


# ════════════════════════════════════════════════════════════════════════════
# PathBundle — shared output contract
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class PathBundle:
    """
    Simulation output container shared by all three process classes.

    Attributes
    ----------
    spots  : np.ndarray, shape (n_paths, n_steps)
             Simulated spot prices in EUR/MWh.
             Column 0 is always S0 (the current spot, identical across paths).

    state2 : np.ndarray or None, shape (n_paths, n_steps)
             Second factor:
               - TwoFactorSchwartzQL : short-term log-deviation χ_t
               - HestonProcessQL     : instantaneous variance v_t
               - LognormalOUProcessQL: None (single-factor model)

    model  : str
             Process identifier: "lognormal_ou" | "schwartz_smith" | "heston"
             Used by BasisFunctions to select appropriate basis terms.

    Notes
    -----
    n_steps equals len(date_grid), i.e. it includes the start date (t=0) so
    spots[:, 0] == S0 for all paths. This matches the convention in
    LognormalOUProcess.simulate() in gas_storage_mc.py.
    """
    spots:  np.ndarray
    state2: Optional[np.ndarray]
    model:  str


# ════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════════════════

def _to_ql_date(d: date) -> ql.Date:
    """Convert Python date to QuantLib Date."""
    return ql.Date(d.day, d.month, d.year)


def _flat_ts(rate: float) -> ql.YieldTermStructureHandle:
    """Build a flat deterministic yield term structure."""
    dc  = ql.Actual365Fixed()
    cal = ql.NullCalendar()
    return ql.YieldTermStructureHandle(
        ql.FlatForward(0, cal, ql.QuoteHandle(ql.SimpleQuote(rate)), dc)
    )


def _build_1d_rsg(
    n_dims:   int,
    seed:     Optional[int],
    use_sobol: bool,
):
    """
    Build a 1-D random sequence generator for single-factor path generators
    (GaussianPathGenerator / GaussianSobolPathGenerator).

    n_dims = n_steps (one random variate per time step).
    """
    if use_sobol:
        uniform_rsg  = ql.UniformLowDiscrepancySequenceGenerator(n_dims)
        gaussian_rsg = ql.GaussianLowDiscrepancySequenceGenerator(uniform_rsg)
    else:
        ql_seed      = (seed + 1) if seed is not None else 42
        uniform_rsg  = ql.UniformRandomSequenceGenerator(
            n_dims, ql.UniformRandomGenerator(ql_seed)
        )
        gaussian_rsg = ql.GaussianRandomSequenceGenerator(uniform_rsg)
    return gaussian_rsg


def _build_nd_rsg(
    n_dims:    int,
    seed:      Optional[int],
    use_sobol: bool,
):
    """
    Build an n-D random sequence generator for multi-factor path generators
    (GaussianMultiPathGenerator / GaussianSobolMultiPathGenerator).

    n_dims = factors × n_steps.
    """
    if use_sobol:
        uniform_rsg  = ql.UniformLowDiscrepancySequenceGenerator(n_dims)
        gaussian_rsg = ql.GaussianLowDiscrepancySequenceGenerator(uniform_rsg)
    else:
        ql_seed      = (seed + 1) if seed is not None else 42
        uniform_rsg  = ql.UniformRandomSequenceGenerator(
            n_dims, ql.UniformRandomGenerator(ql_seed)
        )
        gaussian_rsg = ql.GaussianRandomSequenceGenerator(uniform_rsg)
    return gaussian_rsg


def _extract_1d_paths(
    generator,
    n_paths:  int,
    n_steps:  int,
) -> np.ndarray:
    """
    Draw n_paths from a 1-D path generator and return shape (n_paths, n_steps+1).
    Works with both GaussianPathGenerator and GaussianSobolPathGenerator.
    """
    out = np.empty((n_paths, n_steps + 1))
    for i in range(n_paths):
        path    = generator.next().value()
        out[i]  = [path[j] for j in range(n_steps + 1)]
    return out


def _extract_nd_paths(
    generator,
    n_paths:  int,
    n_steps:  int,
    n_assets: int,
) -> List[np.ndarray]:
    """
    Draw n_paths from a multi-asset path generator.

    Returns a list of n_assets arrays, each shape (n_paths, n_steps+1).
    Works with both GaussianMultiPathGenerator and GaussianSobolMultiPathGenerator.
    """
    arrays = [np.empty((n_paths, n_steps + 1)) for _ in range(n_assets)]
    for i in range(n_paths):
        mp = generator.next().value()   # MultiPath
        for k in range(n_assets):
            path       = mp[k]
            arrays[k][i] = [path[j] for j in range(n_steps + 1)]
    return arrays


# ════════════════════════════════════════════════════════════════════════════
# Step 1 — LognormalOUProcessQL
# ════════════════════════════════════════════════════════════════════════════

class LognormalOUProcessQL:
    """
    Single-factor lognormal Ornstein-Uhlenbeck process via QuantLib.

    Log-prices follow an additive OU around the log-forward curve:

        d(ln S) = κ · (ln F(t) − ln S) · dt  +  σ · dW

    QuantLib backend: ExtendedOrnsteinUhlenbeckProcess.
    The mean-reversion target is a Python callable t → ln F(t), where t is
    measured in years from date_grid[0].

    This is the QuantLib-native equivalent of LognormalOUProcess in
    gas_storage_mc.py.  Both produce the same distribution; this version
    supports Sobol quasi-random sequences for faster convergence.

    Parameters
    ----------
    spot_price     : float        — current spot S0 (EUR/MWh)
    kappa          : float        — mean-reversion speed (yr⁻¹)
    sigma          : float        — log-price annual volatility
    fwd_func       : callable     — date → EUR/MWh, forward curve interpolator
                                   (e.g. MarketParams.forward_price)
    n_paths        : int
    seed           : int or None
    use_sobol      : bool         — Sobol quasi-random if True
    """

    MODEL = "lognormal_ou"

    def __init__(
        self,
        spot_price : float,
        kappa      : float,
        sigma      : float,
        fwd_func,                      # callable: date → EUR/MWh
        n_paths    : int  = 10_000,
        seed       : Optional[int] = 42,
        use_sobol  : bool = False,
    ):
        self.spot_price = spot_price
        self.kappa      = kappa
        self.sigma      = sigma
        self.fwd_func   = fwd_func
        self.n_paths    = n_paths
        self.seed       = seed
        self.use_sobol  = use_sobol

    def simulate(self, date_grid: List[date]) -> PathBundle:
        """
        Simulate price paths on the given date grid.

        Returns PathBundle with:
          spots  : (n_paths, n_steps), EUR/MWh
          state2 : None  (single-factor model)
          model  : "lognormal_ou"
        """
        n_steps  = len(date_grid) - 1
        ref_date = date_grid[0]
        T        = (date_grid[-1] - ref_date).days / 365.25

        # Set QuantLib evaluation date
        ql.Settings.instance().evaluationDate = _to_ql_date(ref_date)

        # Mean-reversion target: t (years from ref_date) → ln F(t)
        # QuantLib calls this at each time step during path generation
        def log_fwd(t_years: float) -> float:
            d = ref_date + __import__('datetime').timedelta(days=round(t_years * 365.25))
            return np.log(self.fwd_func(d))

        process = ql.ExtendedOrnsteinUhlenbeckProcess(
            self.kappa,          # speed κ
            self.sigma,          # σ
            np.log(self.spot_price),   # x0 = ln S0
            log_fwd,             # time-varying mean-reversion level
        )

        time_grid = ql.TimeGrid(T, n_steps)
        rsg       = _build_1d_rsg(n_steps, self.seed, self.use_sobol)

        if self.use_sobol:
            generator = ql.GaussianSobolPathGenerator(
                process, time_grid, rsg, False
            )
        else:
            generator = ql.GaussianPathGenerator(
                process, time_grid, rsg, False
            )

        log_paths = _extract_1d_paths(generator, self.n_paths, n_steps)
        spots     = np.exp(log_paths)   # log-price → EUR/MWh

        return PathBundle(spots=spots, state2=None, model=self.MODEL)


# ════════════════════════════════════════════════════════════════════════════
# Step 2 — TwoFactorSchwartzQL
# ════════════════════════════════════════════════════════════════════════════

class TwoFactorSchwartzQL:
    """
    Two-factor Schwartz-Smith (2000) process via QuantLib.

    The spot price decomposes into a short-term deviation and a long-term
    equilibrium factor:

        ln S_t = χ_t + η_t

    where:
        dχ_t = −κ_χ · χ_t · dt  +  σ_χ · dW₁          (mean-reverting)
        dη_t =  μ_η          · dt  +  σ_η · dW₂          (arithmetic BM, no drift needed here)

        Corr(dW₁, dW₂) = ρ

    This is the correct seasonal gas storage model: χ_t captures the
    summer/winter spread (fast mean-reversion, κ_χ ≈ 1–3 yr⁻¹) while η_t
    captures the long-run structural level (slow drift, κ_η → 0).

    QuantLib backend: StochasticProcessArray wrapping two 1-D processes:
      - Factor 1 (χ): ExtendedOrnsteinUhlenbeckProcess, zero mean level
      - Factor 2 (η): OrnsteinUhlenbeckProcess with κ=0 (approximated as
                      a Brownian motion with small κ_eta)
    Correlated Brownian motions are handled by StochasticProcessArray with
    a 2×2 correlation matrix.
    Path generation uses GaussianMultiPathGenerator.

    The spot price is reconstructed as S_t = exp(χ_t + η_t).

    Initial state decomposition
    ---------------------------
    Given current spot S0 and a long-term forward F_LT:
        η0 = ln(F_LT)                      — long-run equilibrium
        χ0 = ln(S0) − η0                   — short-term deviation

    Parameters
    ----------
    spot_price  : float  — S0 (EUR/MWh)
    eta_0       : float  — initial long-term equilibrium level (EUR/MWh)
                           typically the last pillar of the forward curve
    kappa_xi    : float  — short-term mean-reversion speed (yr⁻¹), e.g. 2.0
    sigma_xi    : float  — short-term factor volatility, e.g. 0.40
    kappa_eta   : float  — long-term factor mean-reversion (≈ 0 for random walk)
                           set to a small positive value, e.g. 0.01
    sigma_eta   : float  — long-term factor volatility, e.g. 0.20
    rho         : float  — correlation dW₁·dW₂, e.g. −0.30 to +0.30
    n_paths     : int
    seed        : int or None
    use_sobol   : bool
    """

    MODEL = "schwartz_smith"

    def __init__(
        self,
        spot_price : float,
        eta_0      : float,
        kappa_xi   : float,
        sigma_xi   : float,
        kappa_eta  : float,
        sigma_eta  : float,
        rho        : float,
        n_paths    : int  = 10_000,
        seed       : Optional[int] = 42,
        use_sobol  : bool = False,
    ):
        self.spot_price = spot_price
        self.eta_0      = eta_0
        self.kappa_xi   = kappa_xi
        self.sigma_xi   = sigma_xi
        self.kappa_eta  = kappa_eta
        self.sigma_eta  = sigma_eta
        self.rho        = rho
        self.n_paths    = n_paths
        self.seed       = seed
        self.use_sobol  = use_sobol

        # Decompose initial state
        self.ln_s0  = np.log(spot_price)
        self.eta0   = np.log(eta_0)            # long-term log-level
        self.xi0    = self.ln_s0 - self.eta0   # short-term log-deviation

    def simulate(self, date_grid: List[date]) -> PathBundle:
        """
        Simulate two-factor paths on the given date grid.

        Returns PathBundle with:
          spots  : (n_paths, n_steps), EUR/MWh — S_t = exp(χ_t + η_t)
          state2 : (n_paths, n_steps), χ_t paths (short-term factor in log-space)
          model  : "schwartz_smith"

        Notes
        -----
        Storing χ_t in state2 is deliberate: it is the volatile, mean-reverting
        factor that drives optionality and should appear in LSMC basis functions.
        η_t can always be recovered as ln(S_t) − χ_t.
        """
        n_steps  = len(date_grid) - 1
        ref_date = date_grid[0]
        T        = (date_grid[-1] - ref_date).days / 365.25

        ql.Settings.instance().evaluationDate = _to_ql_date(ref_date)

        # ── Factor 1: χ_t — mean-reverting short-term deviation ─────────────
        # ExtendedOrnsteinUhlenbeckProcess with zero mean level (mean-reverts to 0)
        proc_xi = ql.ExtendedOrnsteinUhlenbeckProcess(
            self.kappa_xi,          # speed κ_χ
            self.sigma_xi,          # σ_χ
            self.xi0,               # x0 = χ0
            lambda t: 0.0,          # zero mean level (deviation reverts to 0)
        )

        # ── Factor 2: η_t — near-random-walk long-term level ────────────────
        # OrnsteinUhlenbeckProcess with very small κ_η approximates a BM.
        # Using kappa_eta=0.0 exactly would make QL's analytical scheme
        # degenerate; a small positive value (e.g. 0.01) is safe.
        proc_eta = ql.OrnsteinUhlenbeckProcess(
            self.kappa_eta,     # speed κ_η (near zero → near random walk)
            self.sigma_eta,     # σ_η
            self.eta0,          # x0 = η0 = ln(F_LT)
        )

        # ── StochasticProcessArray — correlates the two factors ──────────────
        vec = ql.StochasticProcess1DVector()
        vec.append(proc_xi)
        vec.append(proc_eta)

        corr = ql.Matrix(2, 2)
        corr[0][0] =  1.0;      corr[0][1] = self.rho
        corr[1][0] = self.rho;  corr[1][1] =  1.0

        proc_array = ql.StochasticProcessArray(vec, corr)

        # ── Path generator ───────────────────────────────────────────────────
        # n_dims = 2 factors × n_steps
        time_grid = ql.TimeGrid(T, n_steps)
        rsg       = _build_nd_rsg(2 * n_steps, self.seed, self.use_sobol)

        if self.use_sobol:
            generator = ql.GaussianSobolMultiPathGenerator(
                proc_array, time_grid, rsg, False
            )
        else:
            generator = ql.GaussianMultiPathGenerator(
                proc_array, time_grid, rsg, False
            )

        # ── Extract paths ────────────────────────────────────────────────────
        xi_paths, eta_paths = _extract_nd_paths(
            generator, self.n_paths, n_steps, n_assets=2
        )

        # Spot price: S_t = exp(χ_t + η_t)
        spots = np.exp(xi_paths + eta_paths)

        return PathBundle(spots=spots, state2=xi_paths, model=self.MODEL)


# ════════════════════════════════════════════════════════════════════════════
# Step 3 — HestonProcessQL
# ════════════════════════════════════════════════════════════════════════════

class HestonProcessQL:
    """
    Heston (1993) stochastic volatility process via QuantLib.

    The spot and instantaneous variance follow:

        dS_t  =  r · S_t · dt  +  √v_t · S_t · dW₁
        dv_t  =  κ · (θ − v_t) · dt  +  ξ · √v_t · dW₂

        Corr(dW₁, dW₂) = ρ

    Feller condition for strictly positive variance: 2κθ > ξ².
    If this is violated, v_t can hit zero; the QE martingale discretisation
    handles this gracefully (default in QL).

    Relevance for gas storage
    -------------------------
    Heston is not the primary gas storage model (the two-factor Schwartz-Smith
    model with seasonal mean reversion is more appropriate for spread optionality).
    Heston is included for:
      - Options on storage capacity / tolling agreements
      - Comparison against the implied vol surface
      - LSMC testing with stochastic vol basis functions

    QuantLib backend: HestonProcess with GaussianMultiPathGenerator.
    The process outputs two asset paths:
      - path[0] : spot price S_t
      - path[1] : instantaneous variance v_t

    Parameters
    ----------
    spot_price   : float — S0 (EUR/MWh)
    risk_free_rate: float — continuous risk-free rate r
    v0           : float — initial variance (σ²), e.g. 0.09 for σ=30%
    kappa        : float — variance mean-reversion speed, e.g. 2.0
    theta        : float — long-run variance, e.g. 0.09
    xi           : float — vol-of-vol ξ, e.g. 0.40
    rho          : float — correlation dW₁·dW₂, typically negative, e.g. −0.60
    n_paths      : int
    seed         : int or None
    use_sobol    : bool
    """

    MODEL = "heston"

    def __init__(
        self,
        spot_price     : float,
        risk_free_rate : float,
        v0             : float,
        kappa          : float,
        theta          : float,
        xi             : float,
        rho            : float,
        n_paths        : int  = 10_000,
        seed           : Optional[int] = 42,
        use_sobol      : bool = False,
    ):
        self.spot_price     = spot_price
        self.risk_free_rate = risk_free_rate
        self.v0             = v0
        self.kappa          = kappa
        self.theta          = theta
        self.xi             = xi
        self.rho            = rho
        self.n_paths        = n_paths
        self.seed           = seed
        self.use_sobol      = use_sobol

        # Feller condition check (informational)
        feller = 2.0 * kappa * theta - xi ** 2
        if feller <= 0:
            import warnings
            warnings.warn(
                f"Heston Feller condition violated: 2κθ − ξ² = {feller:.4f} ≤ 0. "
                "Variance can hit zero. QE discretisation handles this, but "
                "calibrate carefully.",
                stacklevel=2,
            )

    def simulate(self, date_grid: List[date]) -> PathBundle:
        """
        Simulate Heston paths on the given date grid.

        Returns PathBundle with:
          spots  : (n_paths, n_steps), EUR/MWh — S_t
          state2 : (n_paths, n_steps), v_t paths (instantaneous variance)
          model  : "heston"

        Notes
        -----
        The QuantLib HestonProcess uses the Quadratic Exponential Martingale
        discretisation by default, which is the most accurate scheme for Heston
        and preserves the martingale property of the discounted spot price.

        state2 carries v_t (not σ_t = √v_t) to match the basis function
        convention where variance enters linearly in the regression.
        """
        n_steps  = len(date_grid) - 1
        ref_date = date_grid[0]
        T        = (date_grid[-1] - ref_date).days / 365.25

        ql.Settings.instance().evaluationDate = _to_ql_date(ref_date)

        # ── QuantLib yield term structures ───────────────────────────────────
        rf_ts  = _flat_ts(self.risk_free_rate)
        div_ts = _flat_ts(0.0)   # no dividend / convenience yield here;
                                 # for gas storage the convenience yield is
                                 # implicit in the storage value, not the process
        s0 = ql.QuoteHandle(ql.SimpleQuote(self.spot_price))

        # ── HestonProcess ────────────────────────────────────────────────────
        process = ql.HestonProcess(
            rf_ts, div_ts, s0,
            self.v0,    # initial variance
            self.kappa, # mean-reversion speed
            self.theta, # long-run variance
            self.xi,    # vol-of-vol
            self.rho,   # correlation S-v
            ql.HestonProcess.QuadraticExponentialMartingale,
        )

        # ── Path generator ───────────────────────────────────────────────────
        # HestonProcess is a 2-factor process: n_dims = 2 × n_steps
        time_grid = ql.TimeGrid(T, n_steps)
        rsg       = _build_nd_rsg(2 * n_steps, self.seed, self.use_sobol)

        if self.use_sobol:
            generator = ql.GaussianSobolMultiPathGenerator(
                process, time_grid, rsg, False
            )
        else:
            generator = ql.GaussianMultiPathGenerator(
                process, time_grid, rsg, False
            )

        # ── Extract paths ────────────────────────────────────────────────────
        # path[0] = spot, path[1] = variance
        spot_paths, var_paths = _extract_nd_paths(
            generator, self.n_paths, n_steps, n_assets=2
        )

        return PathBundle(spots=spot_paths, state2=var_paths, model=self.MODEL)


# ════════════════════════════════════════════════════════════════════════════
# Quick smoke test
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import config as cfg
    from gas_storage_mc import _build_date_grid, MarketParams

    market    = MarketParams(**cfg.MARKET)
    date_grid = _build_date_grid(cfg.CALENDAR["start_date"], cfg.CALENDAR["end_date"])
    n_paths   = cfg.SIMULATION["n_paths"]
    seed      = cfg.SIMULATION["seed"]

    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  processes_ql.py — Smoke Test (3 process classes)           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"  date_grid : {date_grid[0]} → {date_grid[-1]}  ({len(date_grid)} steps)")
    print(f"  n_paths   : {n_paths:,}")
    print()

    # ── 1. LognormalOUProcessQL ──────────────────────────────────────────────
    print("[ 1 / 3 ]  LognormalOUProcessQL")
    proc_ou = LognormalOUProcessQL(
        spot_price = market.spot_price,
        kappa      = market.kappa,
        sigma      = market.sigma,
        fwd_func   = market.forward_price,
        n_paths    = n_paths,
        seed       = seed,
        use_sobol  = False,
    )
    bundle_ou = proc_ou.simulate(date_grid)
    print(f"  spots shape   : {bundle_ou.spots.shape}")
    print(f"  state2        : {bundle_ou.state2}")
    print(f"  model         : {bundle_ou.model!r}")
    print(f"  S0 (col 0)    : {bundle_ou.spots[:, 0].mean():.4f}  (expect {market.spot_price})")
    print(f"  mean terminal : {bundle_ou.spots[:, -1].mean():.4f}")
    print()

    # ── 2. TwoFactorSchwartzQL ───────────────────────────────────────────────
    print("[ 2 / 3 ]  TwoFactorSchwartzQL")
    # eta_0: long-term equilibrium ≈ last pillar of the forward curve
    fwd_pillars = sorted(market.forward_curve.items())
    eta_0_price = fwd_pillars[-1][1]   # last forward pillar as long-run level

    proc_ss = TwoFactorSchwartzQL(
        spot_price = market.spot_price,
        eta_0      = eta_0_price,
        kappa_xi   = 2.0,      # fast short-term reversion (yr⁻¹)
        sigma_xi   = 0.35,     # short-term vol
        kappa_eta  = 0.01,     # near random-walk long-term factor
        sigma_eta  = 0.15,     # long-term vol
        rho        = -0.20,    # mild negative correlation
        n_paths    = n_paths,
        seed       = seed,
    )
    bundle_ss = proc_ss.simulate(date_grid)
    print(f"  spots shape   : {bundle_ss.spots.shape}")
    print(f"  state2 shape  : {bundle_ss.state2.shape}  (χ_t paths)")
    print(f"  model         : {bundle_ss.model!r}")
    print(f"  S0 (col 0)    : {bundle_ss.spots[:, 0].mean():.4f}  (expect {market.spot_price:.4f})")
    print(f"  mean terminal : {bundle_ss.spots[:, -1].mean():.4f}")
    print(f"  χ mean @ end  : {bundle_ss.state2[:, -1].mean():.4f}  (expect ~0)")
    print()

    # ── 3. HestonProcessQL ──────────────────────────────────────────────────
    print("[ 3 / 3 ]  HestonProcessQL")
    proc_h = HestonProcessQL(
        spot_price     = market.spot_price,
        risk_free_rate = market.risk_free_rate,
        v0             = market.sigma ** 2,  # initial variance = σ²
        kappa          = 2.0,
        theta          = market.sigma ** 2,  # long-run variance = σ²
        xi             = 0.40,               # vol-of-vol
        rho            = -0.60,
        n_paths        = n_paths,
        seed           = seed,
    )
    bundle_h = proc_h.simulate(date_grid)
    print(f"  spots shape   : {bundle_h.spots.shape}")
    print(f"  state2 shape  : {bundle_h.state2.shape}  (v_t paths)")
    print(f"  model         : {bundle_h.model!r}")
    print(f"  S0 (col 0)    : {bundle_h.spots[:, 0].mean():.4f}  (expect {market.spot_price})")
    print(f"  mean terminal : {bundle_h.spots[:, -1].mean():.4f}")
    print(f"  v0 (col 0)    : {bundle_h.state2[:, 0].mean():.6f}  (expect {market.sigma**2:.6f})")
    print(f"  mean var @ end: {bundle_h.state2[:, -1].mean():.6f}")
    print()
    print("Smoke test complete.")
