"""
lsmc.py — Least-Squares Monte Carlo (LSMC) Engine for Gas Storage Valuation
=============================================================================
Implements the Longstaff-Schwartz (2001) backward-induction algorithm adapted
for physical gas storage constraints, following the approach used on
professional energy trading desks.

Architecture
------------
This module is self-contained and depends only on:
    gas_storage_mc  — StorageParams, MarketParams, _discount_factor, _build_date_grid
    processes       — PathBundle (the common path output contract)

It does NOT depend on StorageDispatcher, LognormalOUProcess, or any other
engine from gas_storage_mc.  The LSMC engine plugs in alongside the existing
greedy-MC and LP-intrinsic pipelines; all three run independently and their
results are compared in GasStorageSimulator.run().

Module structure
----------------
    LSMCParams          dataclass — algorithm hyperparameters (from config.LSMC)
    BasisFunctions      class     — builds regression design matrix X(x, bundle, t)
    LSMCOptimiser       class     — backward induction fitting pass → LSMCPolicy
    LSMCPolicy          class     — output of fitting pass (coefficients + diagnostics)
    LSMCResult          class     — output of forward pricing pass (NPV array)

LSMC algorithm — corrected value-iteration backward pass
---------------------------------------------------------

Phase 1 (forward sweep — inventory generation):
    Run a perturbed-greedy forward pass on the fit paths to generate
    path-consistent inventory trajectories with good cross-sectional spread.
    At each step, the greedy action (inject when price rising, withdraw when
    falling) is randomly flipped for ~50% of paths, producing diverse
    inventory states across [min_inventory, max_inventory] at every time step.
    This is critical: without spread, the regression has no inventory signal.

Phase 2 (backward regression — value iteration):
    Starting from the terminal state value V_T = x_T · S_T, step backward:

    For t = T-1 down to 0:
        1.  Regress V_{t+1}(x_{t+1}, S_{t+1}) onto basis(x_{t+1}, S_{t+1}):
                β_t = argmin || X_{t+1} · β - V_{t+1} ||²  (with ridge penalty)
            This fits the VALUE FUNCTION at t+1 as a function of next-step state.

        2.  For each candidate action a (n_actions evenly spaced volumes):
                Q(a) = CF_t(a, S_t) + df · β_t · basis(x_t + a, S_{t+1})
            where the continuation is evaluated at the NEXT state (x_t + a, S_{t+1}).

        3.  Optimal action:  a*(t) = argmax_a Q(a)   (feasibility constrained)

        4.  Value-iteration update:
                V_t(x_t, S_t) = max_a Q(a)  =  best_val
            CRITICAL: we use best_val (the regression-estimated optimal value)
            as V_t — NOT CF(a*) + df · V_{t+1}(actual).
            This propagates the regression approximation backward cleanly,
            avoiding the accumulation of policy errors from wrong-trajectory CFs.

        5.  Store β_t in LSMCPolicy.coefficients[t].

Forward pass (pricing):
    t = 0 to T-1:
        Using stored β_t, apply the same argmax action selection on the
        out-of-sample pricing paths. Accumulate discounted cash flows.
    Terminal: remaining inventory liquidated at S_T (with constraint penalty).
    Return NPV per path.

Key design decisions
--------------------
Regression features at (x_{t+1}, S_{t+1}) not (x_t, S_t):
    The regression fits V_{t+1} as a function of the state at t+1.
    When selecting actions, we evaluate this function at (x_t + a, S_{t+1}).
    This is consistent: same features for fitting and evaluation.
    Using step-t features for a value that lives at t+1 creates extrapolation
    errors that corrupt the policy.

Value-iteration update (cont_val = best_val):
    The continuation value passed to the regression at step t-1 should be
    the OPTIMAL VALUE at step t, not the CF of a specific (possibly wrong) policy.
    Using best_val = max_a Q(a) ensures the backward pass propagates the
    regression-estimated optimal value, which is consistent with the fitting target.
    The alternative (CF(a*) + df · cont_val) accumulates errors from the
    perturbed-greedy training trajectory, causing the continuation values to drift
    to large negative numbers over 365 steps.

Perturbed-greedy inventory for training:
    A plain greedy heuristic (inject when price rising) produces near-degenerate
    inventory distributions — all paths converge to the same level at each step
    (near-zero in winter, near-full in summer) because prices are correlated.
    Adding random action flips for ~50% of paths spreads inventory across the
    full feasible range while maintaining path consistency (trajectories still
    respect rate and capacity constraints). This is essential for the regression
    to capture the inventory dimension of the continuation value.

Basis functions
---------------
For each model (identified by bundle.model):

    lognormal_ou :
        [1,  S̃,  S̃²,  S̃³,  x̃,  x̃²,  x̃³,  x̃·S̃]
        where S̃ = normalised spot, x̃ = normalised inventory

    schwartz_smith :
        [1,  S̃,  S̃²,  S̃³,  χ̃,  χ̃²,  χ̃³,  x̃,  x̃²,  x̃³,  x̃·S̃,  x̃·χ̃]
        χ̃ = normalised short-term factor from bundle.state2

    heston :
        [1,  S̃,  S̃²,  S̃³,  ṽ,  ṽ²,  x̃,  x̃²,  x̃³,  x̃·S̃,  x̃·ṽ]
        ṽ = normalised instantaneous variance from bundle.state2

Normalisation: each column is standardised to zero mean and unit std using
statistics from the fit paths.  This is critical for numerical stability of
the least-squares regression, particularly when spot prices are O(35) and
inventories are O(1e6) — without normalisation, the design matrix is poorly
conditioned and regression coefficients are meaningless.

Action discretisation
---------------------
At each step, n_actions evenly spaced volume candidates are tested:
    {1/n, 2/n, ..., n/n} × max_rate  (plus idle = 0)
for both injection and withdrawal independently.  The feasible set is
intersected with inventory constraints.  After argmax selection, no Newton
refinement step is applied — the action grid is sufficient for the 5-level
discretisation used here; finer grids (n_actions=10) can be used for
production without any code change.

In-sample / out-of-sample split
--------------------------------
fit_paths (from LSMCParams) paths are used for the backward fitting pass.
The remaining n_paths - fit_paths paths are used for the forward pricing pass.
Setting fit_paths = n_paths uses all paths for both passes (introduces a small
upward bias; acceptable when n_paths >= 10,000 and poly_degree <= 3).

References
----------
Longstaff & Schwartz (2001) — "Valuing American Options by Simulation"
Bjerksund, Stensland & Vagstad (2011) — "Gas Storage Valuation: Price
    Modelling v. Optimization Methods"
Andersen & Broadie (2004) — "Primal-Dual Simulation Algorithm for Pricing
    Multidimensional American Options"
Thompson et al. (2009) — "Natural Gas Storage Valuation, Optimisation,
    Market and Credit Risk Management"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from gas_storage_mc import StorageParams, MarketParams
    from processes import PathBundle


# ════════════════════════════════════════════════════════════════════════════
# LSMCParams
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class LSMCParams:
    """
    LSMC algorithm hyperparameters.  Constructed from config.LSMC in
    run_simulation.build_params().  All fields required — no defaults,
    consistent with project-wide dataclass convention.

    Fields
    ------
    n_actions    : int
        Number of volume candidates per action type (inject / withdraw).
        Candidates are evenly spaced from 1/n to 100% of max rate.
        n_actions=5 → {20%, 40%, 60%, 80%, 100%} of max rate plus idle (0%).
        Increase to 10 for higher accuracy at a proportional runtime cost.

    poly_degree  : int
        Polynomial degree for price and inventory basis terms.
        2 = quadratic (fast, sometimes underfits near constraints).
        3 = cubic (recommended; captures asymmetric boundary behaviour).

    basis_type   : str
        Polynomial family for price terms.
        "power"     — standard monomials [S, S², S³]  (default)
        "laguerre"  — Laguerre polynomials (original LS choice; non-negative domain)
        "chebyshev" — Chebyshev T polynomials (better conditioning on bounded domain)

    cross_terms  : bool
        Include inventory × price (and inventory × state2) cross terms.
        These capture the interaction between inventory level and price
        optionality and are strongly recommended for storage valuation.

    fit_paths    : int
        Number of paths used for the backward regression fitting pass.
        Remaining paths (n_paths - fit_paths) are used for forward pricing.
        Set equal to n_paths to use all paths for both passes (minor upward
        bias; acceptable for n_paths >= 10,000).

    regularise   : float
        Ridge (L2) regularisation coefficient added to the diagonal of X'X
        before solving the least-squares system.  Prevents ill-conditioned
        regressions when many basis functions are used on a small path count.
        Typical range: 1e-6 to 1e-3.  Set to 0.0 to disable.
    """
    n_actions   : int
    poly_degree : int
    basis_type  : str
    cross_terms : bool
    fit_paths   : int
    regularise  : float


# ════════════════════════════════════════════════════════════════════════════
# BasisFunctions
# ════════════════════════════════════════════════════════════════════════════

class BasisFunctions:
    """
    Build the regression design matrix X of shape (n_paths, n_basis) at each
    time step, using normalised spot price, inventory, and (for multi-factor
    models) the second state variable.

    The normalisation statistics (mean, std) are computed once from the fit
    paths at the beginning of the backward pass and stored as instance
    attributes.  They are reused during the forward pricing pass so that the
    forward paths are projected into the same normalised space as the fitted
    coefficients.

    Usage
    -----
    basis = BasisFunctions(lsmc_params, storage)
    basis.fit_normalisation(spots_fit, inventories_fit, state2_fit)
    X = basis.evaluate(inventory_t, spots_t, state2_t, model)   # (n_fit, n_basis)
    """

    def __init__(self, params: LSMCParams, storage):
        self.params  = params
        self.storage = storage
        # Normalisation statistics — set by fit_normalisation()
        self._s_mean  = 0.0;  self._s_std  = 1.0
        self._x_mean  = 0.0;  self._x_std  = 1.0
        self._z_mean  = 0.0;  self._z_std  = 1.0   # state2 (χ or v)
        self._fitted  = False

    def fit_normalisation(
        self,
        spots:       np.ndarray,          # (n_fit, n_steps+1) — fit paths only
        inventories: np.ndarray,          # (n_fit, n_steps+1)
        state2:      Optional[np.ndarray],  # (n_fit, n_steps+1) or None
    ) -> None:
        """
        Compute and store normalisation statistics from the fit paths.
        Called once before the backward pass begins.
        Uses all time steps to get stable global statistics.
        """
        self._s_mean = spots.mean();        self._s_std  = max(spots.std(), 1e-8)
        self._x_mean = inventories.mean();  self._x_std  = max(inventories.std(), 1e-8)
        if state2 is not None:
            self._z_mean = state2.mean();   self._z_std  = max(state2.std(), 1e-8)
        self._fitted = True

    def evaluate(
        self,
        inventory_t: np.ndarray,          # (n_paths,) — inventory
        spots_t:     np.ndarray,          # (n_paths,) — spot price
        state2_t:    Optional[np.ndarray],  # (n_paths,) or None
        model:       str,
    ) -> np.ndarray:
        """
        Return design matrix X of shape (n_paths, n_basis).

        Columns depend on model and params:
            lognormal_ou   : [1, S̃, S̃², ..., x̃, x̃², ..., (x̃·S̃)]
            schwartz_smith : [1, S̃, ..., χ̃, χ̃², ..., x̃, x̃², ..., (x̃·S̃), (x̃·χ̃)]
            heston         : [1, S̃, ..., ṽ, ṽ², x̃, x̃², ..., (x̃·S̃), (x̃·ṽ)]
        """
        S̃ = (spots_t      - self._s_mean) / self._s_std
        x̃ = (inventory_t  - self._x_mean) / self._x_std

        cols = [np.ones(len(S̃))]                          # intercept

        # Price polynomial terms
        cols += self._poly_terms(S̃)

        # Second state variable terms (Schwartz-Smith or Heston)
        z̃ = None
        if state2_t is not None and model in ("schwartz_smith", "heston"):
            z̃ = (state2_t - self._z_mean) / self._z_std
            cols += self._poly_terms(z̃)

        # Inventory polynomial terms
        cols += self._poly_terms(x̃)

        # Cross terms
        if self.params.cross_terms:
            cols.append(x̃ * S̃)
            if z̃ is not None:
                cols.append(x̃ * z̃)

        return np.column_stack(cols)  # (n_paths, n_basis)

    def _poly_terms(self, z: np.ndarray) -> List[np.ndarray]:
        """
        Return list of polynomial basis vectors for scalar z (normalised).
        Degree is controlled by params.poly_degree.
        Basis family is controlled by params.basis_type.
        """
        d = self.params.poly_degree
        t = self.params.basis_type

        if t == "laguerre":
            # Laguerre polynomials L_1, ..., L_d
            from numpy.polynomial.laguerre import lagval
            return [lagval(z, [0] * k + [1]) for k in range(1, d + 1)]

        elif t == "chebyshev":
            # Chebyshev T polynomials T_1, ..., T_d
            from numpy.polynomial.chebyshev import chebval
            return [chebval(np.clip(z, -1, 1), [0] * k + [1]) for k in range(1, d + 1)]

        else:  # "power" — standard monomials
            return [z ** k for k in range(1, d + 1)]


# ════════════════════════════════════════════════════════════════════════════
# LSMCPolicy  (output of fitting pass, input of pricing pass)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class LSMCPolicy:
    """
    Stores the fitted regression coefficients from the backward induction pass.

    coefficients : List[np.ndarray], length n_steps
        coefficients[t] has shape (n_basis,).
        At each step t, the estimated continuation value for a path with
        next-step feature vector φ is: Ĉ_{t+1} = coefficients[t] @ φ.
        (Note: features are evaluated at t+1 state, consistent with how
        the regression was fitted.)

    r2_scores    : np.ndarray, shape (n_steps,)
        In-sample R² of the regression at each time step.
        Values above 0.90 indicate good basis coverage.
        Low R² at the earliest steps (t < 30) is normal because inventory
        has not yet diverged from the common starting point.

    basis        : BasisFunctions
        The fitted basis instance (carries normalisation statistics).

    model        : str
        The PathBundle.model string — used by the forward pass to select
        the correct basis columns.
    """
    coefficients : List[np.ndarray]  # length n_steps, each (n_basis,)
    r2_scores    : np.ndarray        # (n_steps,)
    basis        : BasisFunctions
    model        : str


# ════════════════════════════════════════════════════════════════════════════
# LSMCResult  (output of forward pricing pass)
# ════════════════════════════════════════════════════════════════════════════

class LSMCResult:
    """
    LSMC valuation output — NPV distribution and diagnostics.

    Attributes
    ----------
    npvs         : np.ndarray, shape (n_price_paths,)
                   NPV per pricing path (EUR).
    policy       : LSMCPolicy — fitted coefficients + R² diagnostics.
    date_grid    : List[date]
    """

    def __init__(
        self,
        npvs      : np.ndarray,
        policy    : LSMCPolicy,
        date_grid : List[date],
    ):
        self.npvs      = npvs
        self.policy    = policy
        self.date_grid = date_grid

    @property
    def mean_npv(self) -> float:
        return float(self.npvs.mean())

    @property
    def std_npv(self) -> float:
        return float(self.npvs.std())

    def percentile(self, p: float) -> float:
        return float(np.percentile(self.npvs, p))

    def r2_summary(self) -> str:
        """Return a one-line R² summary across all time steps."""
        r2 = self.policy.r2_scores
        n_low = (r2 < 0.30).sum()
        return (
            f"Regression R²  mean={r2.mean():.3f}  "
            f"min={r2.min():.3f}  max={r2.max():.3f}  "
            f"steps<0.30: {n_low}/{len(r2)}"
        )

    def summary(self) -> str:
        lines = [
            "=" * 55,
            "  LSMC Valuation Results",
            "=" * 55,
            f"  Mean NPV (LSMC)    : EUR {self.mean_npv:>12,.0f}",
            f"  Std  NPV (LSMC)    : EUR {self.std_npv:>12,.0f}",
            f"  P05                : EUR {self.percentile(5):>12,.0f}",
            f"  P50                : EUR {self.percentile(50):>12,.0f}",
            f"  P95                : EUR {self.percentile(95):>12,.0f}",
            f"  {self.r2_summary()}",
            "=" * 55,
        ]
        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# LSMCOptimiser  —  backward fitting pass  →  LSMCPolicy
# ════════════════════════════════════════════════════════════════════════════

class LSMCOptimiser:
    """
    Backward induction engine.

    Usage
    -----
    optimiser = LSMCOptimiser(storage, market, date_grid, lsmc_params)
    policy    = optimiser.fit(bundle)            # backward pass → LSMCPolicy
    result    = optimiser.price(bundle, policy)  # forward pass  → LSMCResult
    """

    def __init__(
        self,
        storage   : StorageParams,
        market    : MarketParams,
        date_grid : List[date],
        params    : LSMCParams,
    ):
        self.storage   = storage
        self.market    = market
        self.date_grid = date_grid
        self.params    = params
        self.n_steps   = len(date_grid) - 1

        # Pre-compute daily discount factors from t=0
        t0 = date_grid[0]
        self._df = np.array([
            np.exp(-market.risk_free_rate * (d - t0).days / 365.25)
            for d in date_grid
        ])

        # Pre-compute max rates for each step
        self._max_inj = np.array([
            storage.max_injection_rate(date_grid[i].month)
            for i in range(self.n_steps)
        ])
        self._max_wit = np.array([
            storage.max_withdrawal_rate(date_grid[i].month)
            for i in range(self.n_steps)
        ])

    # ── public API ──────────────────────────────────────────────────────────

    def fit(self, bundle: PathBundle) -> LSMCPolicy:
        """
        Backward induction fitting pass — two-phase value-iteration approach.

        Phase 1 — Perturbed-greedy forward sweep:
            Generate diverse inventory trajectories for the fit paths by
            running a greedy heuristic (inject when price rising, withdraw
            when falling) with 50% random action flips.  This produces
            path-consistent inventory states with good cross-sectional spread
            across [min_inventory, max_inventory] at every time step.
            Pure greedy produces near-degenerate states (all paths converge
            to the same level) due to correlated OU price paths.

        Phase 2 — Value-iteration backward regression:
            For t = T-1 down to 0:
            1.  Regress V_{t+1}(x_{t+1}, S_{t+1}) on basis(x_{t+1}, S_{t+1}).
            2.  Find a*(t) = argmax_a [CF_t(a) + df · β · basis(x_t+a, S_{t+1})].
            3.  Set V_t = best_val = max_a Q(a)  (value-iteration update).
                This is the regression-estimated optimal value at state (x_t, S_t).
                Using best_val (not CF(a*) + df · actual_cont_val) prevents
                policy errors from accumulating over 365 backward steps.

        Parameters
        ----------
        bundle : PathBundle — full path set from process.simulate()

        Returns
        -------
        LSMCPolicy — fitted coefficients and R² diagnostics.
        """
        n_fit    = self.params.fit_paths
        spots    = bundle.spots[:n_fit]       # (n_fit, n_steps+1)
        state2   = bundle.state2[:n_fit] if bundle.state2 is not None else None
        model    = bundle.model
        n_paths  = spots.shape[0]
        n_steps  = self.n_steps
        st       = self.storage

        # ── Phase 1: perturbed-greedy inventory generation ────────────────
        inventory = self._perturbed_greedy_inventory(spots, model, state2)
        # inventory shape: (n_fit, n_steps+1)
        # Spread across [0, max_inventory] at every step via ~50% random flips.

        # ── Initialise basis normalisation ────────────────────────────────
        # Use all time steps for stable global statistics.
        basis = BasisFunctions(self.params, st)
        basis.fit_normalisation(spots, inventory, state2)

        # ── Terminal state value V_T = x_T · S_T ─────────────────────────
        cont_val = inventory[:, -1] * spots[:, -1]   # (n_fit,)
        # Terminal constraint penalty (soft): penalise inventory outside bounds
        viol_low  = np.maximum(0.0, st.terminal_min_inventory - inventory[:, -1])
        viol_high = np.maximum(0.0, inventory[:, -1] - st.terminal_max_inventory)
        penalty   = (viol_low + viol_high) * spots[:, -1] * 2.0
        cont_val -= penalty

        # ── Phase 2: backward regression with value-iteration update ─────
        coefficients = [None] * n_steps
        r2_scores    = np.zeros(n_steps)

        for t in range(n_steps - 1, -1, -1):
            S_t    = spots[:, t]              # (n_fit,) — current spot
            S_tp1  = spots[:, t + 1]          # (n_fit,) — next spot
            x_t    = inventory[:, t]          # (n_fit,) — current inventory
            x_tp1  = inventory[:, t + 1]      # (n_fit,) — next inventory (Phase 1)
            z_tp1  = state2[:, t + 1] if state2 is not None else None
            df_step = self._df[t + 1] / self._df[t]

            # ── Regression: fit V_{t+1} on basis(x_{t+1}, S_{t+1}) ────────
            # Features at NEXT step so the fitted function can be evaluated
            # at any (x_t+a, S_{t+1}) during action selection.
            X = basis.evaluate(x_tp1, S_tp1, z_tp1, model)   # (n_fit, n_basis)
            XtX = X.T @ X
            if self.params.regularise > 0:
                XtX += self.params.regularise * np.eye(XtX.shape[0])
            beta     = np.linalg.solve(XtX, X.T @ cont_val)
            cont_hat = X @ beta

            ss_tot = ((cont_val - cont_val.mean()) ** 2).sum()
            ss_res = ((cont_val - cont_hat) ** 2).sum()
            r2_scores[t]    = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
            coefficients[t] = beta

            # ── Action selection: argmax CF_t(a) + df · β · basis(x_t+a, S_{t+1}) ──
            best_val = np.full(n_fit, -np.inf)

            s_next_norm = (S_tp1 - basis._s_mean) / basis._s_std
            z_next_norm = None
            if state2 is not None and model in ("schwartz_smith", "heston"):
                z_next_norm = (state2[:, t + 1] - basis._z_mean) / basis._z_std

            for net_vol in self._action_candidates(t):
                x_next   = np.clip(x_t + net_vol, st.min_inventory, st.max_inventory)
                feasible = self._is_feasible(net_vol, x_t, x_next, t)
                cf       = self._immediate_cf(S_t, net_vol)

                x_next_norm = (x_next - basis._x_mean) / basis._x_std
                cont_approx = self._eval_continuation(
                    x_next_norm, s_next_norm, z_next_norm, beta, basis, model
                )
                total = np.where(feasible, cf + df_step * cont_approx, -np.inf)
                better   = total > best_val
                best_val = np.where(better, total, best_val)

            # ── Value-iteration update: V_t = max_a Q(a) ──────────────────
            # IMPORTANT: use best_val (regression-estimated optimal value),
            # NOT CF(a*) + df · cont_val (which accumulates policy errors).
            cont_val = best_val

        return LSMCPolicy(
            coefficients = coefficients,
            r2_scores    = r2_scores,
            basis        = basis,
            model        = model,
        )

    def price(
        self,
        bundle : PathBundle,
        policy : LSMCPolicy,
    ) -> LSMCResult:
        """
        Forward pricing pass using the fitted policy.

        Uses the pricing paths (bundle.spots[fit_paths:]) if a fit/price
        split is configured; otherwise uses all paths.

        For each pricing path and time step, applies the same argmax action
        selection as the backward pass, but uses the stored β_t coefficients
        instead of computing them online.

        Returns
        -------
        LSMCResult with NPV per pricing path.
        """
        n_fit  = self.params.fit_paths
        n_all  = bundle.spots.shape[0]
        # Pricing paths: out-of-sample if split, else full bundle
        if n_fit < n_all:
            spots  = bundle.spots[n_fit:]
            state2 = bundle.state2[n_fit:] if bundle.state2 is not None else None
        else:
            spots  = bundle.spots
            state2 = bundle.state2

        n_price = spots.shape[0]
        n_steps = self.n_steps
        st      = self.storage
        basis   = policy.basis
        model   = policy.model

        # Per-path inventory state
        inventory = np.full(n_price, st.initial_inventory)
        npvs      = np.zeros(n_price)

        for t in range(n_steps):
            S_t   = spots[:, t]
            S_tp1 = spots[:, t + 1]
            z_tp1 = state2[:, t + 1] if state2 is not None else None
            df_t  = self._df[t]

            beta = policy.coefficients[t]   # (n_basis,)

            # Evaluate continuation for all candidate actions, pick best
            best_val = np.full(n_price, -np.inf)
            best_net = np.zeros(n_price)

            df_step = self._df[t + 1] / self._df[t]

            s_next_norm = (S_tp1 - basis._s_mean) / basis._s_std
            z_next_norm = None
            if state2 is not None and model in ("schwartz_smith", "heston"):
                z_next_norm = (z_tp1 - basis._z_mean) / basis._z_std

            for net_vol in self._action_candidates(t):
                x_next = np.clip(inventory + net_vol, st.min_inventory, st.max_inventory)
                feasible = self._is_feasible(net_vol, inventory, x_next, t)
                cf       = self._immediate_cf(S_t, net_vol)

                x_next_norm = (x_next - basis._x_mean) / basis._x_std
                cont_approx = self._eval_continuation(
                    x_next_norm, s_next_norm, z_next_norm, beta, basis, model
                )

                total = np.where(feasible, cf + df_step * cont_approx, -np.inf)
                better = total > best_val
                best_val = np.where(better, total, best_val)
                best_net = np.where(better, net_vol, best_net)

            # Apply best action
            cf_best   = self._immediate_cf_vec(S_t, best_net)
            npvs     += df_t * cf_best
            inventory = np.clip(inventory + best_net, st.min_inventory, st.max_inventory)

        # Terminal: liquidate remaining inventory
        df_T     = self._df[-1]
        S_T      = spots[:, -1]
        viol_lo  = np.maximum(0.0, st.terminal_min_inventory - inventory)
        viol_hi  = np.maximum(0.0, inventory - st.terminal_max_inventory)
        penalty  = (viol_lo + viol_hi) * S_T * 2.0
        npvs    += df_T * (inventory * S_T - penalty)

        return LSMCResult(npvs=npvs, policy=policy, date_grid=self.date_grid)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _action_candidates(self, t: int) -> List[float]:
        """
        Generate candidate net volumes for step t.

        Returns a list of n_actions inject candidates (positive),
        n_actions withdraw candidates (negative), and 0 (idle).
        Rates are taken from the pre-computed _max_inj / _max_wit arrays.
        """
        n  = self.params.n_actions
        inj_max = self._max_inj[t]
        wit_max = self._max_wit[t]
        candidates = [0.0]
        if inj_max > 0:
            candidates += [inj_max * k / n for k in range(1, n + 1)]
        if wit_max > 0:
            candidates += [-wit_max * k / n for k in range(1, n + 1)]
        return candidates

    def _is_feasible(
        self,
        net_vol  : float,
        x_t      : np.ndarray,
        x_next   : np.ndarray,
        t        : int,
    ) -> np.ndarray:
        """
        Boolean (n_paths,) mask — True where this action is physically feasible.
        Constraints:
          • Injection rate ≤ max_injection_rate
          • Withdrawal rate ≤ max_withdrawal_rate
          • Resulting inventory within [min_inventory, max_inventory]
        """
        min_inv = self.storage.min_inventory
        max_inv = self.storage.max_inventory
        inv_ok  = (x_next >= min_inv) & (x_next <= max_inv)

        if net_vol > 0:   # injection
            rate_ok = net_vol <= self._max_inj[t]
        elif net_vol < 0:  # withdrawal
            rate_ok = abs(net_vol) <= self._max_wit[t]
            # Cannot withdraw more than current inventory
            inv_ok  = inv_ok & (x_t + net_vol >= min_inv)
        else:
            rate_ok = True

        return inv_ok & rate_ok

    def _immediate_cf(self, S_t: np.ndarray, net_vol: float) -> np.ndarray:
        """
        Immediate undiscounted cash flow (EUR) for a scalar net_vol applied
        to all paths simultaneously.  Vectorised over paths.
        """
        st = self.storage
        if net_vol > 0:   # injection: buy gas
            cf = -(S_t * net_vol + st.injection_cost_per_mwh * net_vol)
        elif net_vol < 0:  # withdrawal: sell gas
            sold = abs(net_vol) * st.withdrawal_efficiency
            cf   = S_t * sold - st.withdrawal_cost_per_mwh * abs(net_vol)
        else:
            cf = np.zeros(len(S_t))
        cf = cf - st.daily_fixed_cost   # fixed cost every day
        return cf

    def _immediate_cf_vec(
        self,
        S_t     : np.ndarray,   # (n_paths,)
        net_vol : np.ndarray,   # (n_paths,) — per-path best action
    ) -> np.ndarray:
        """
        Immediate undiscounted cash flow for a per-path net_vol vector.
        Used in the forward pricing pass where each path may take a different action.
        """
        st  = self.storage
        cf  = np.zeros(len(S_t))
        inj = net_vol > 0
        wit = net_vol < 0

        if inj.any():
            v = net_vol[inj]
            cf[inj] = -(S_t[inj] * v + st.injection_cost_per_mwh * v)

        if wit.any():
            v    = net_vol[wit]
            sold = np.abs(v) * st.withdrawal_efficiency
            cf[wit] = S_t[wit] * sold - st.withdrawal_cost_per_mwh * np.abs(v)

        cf -= st.daily_fixed_cost
        return cf

    def _eval_continuation(
        self,
        x_norm  : np.ndarray,              # (n_paths,) normalised inventory
        s_norm  : np.ndarray,              # (n_paths,) normalised spot
        z_norm  : Optional[np.ndarray],    # (n_paths,) normalised state2 or None
        beta    : np.ndarray,              # (n_basis,) fitted coefficients
        basis   : BasisFunctions,
        model   : str,
    ) -> np.ndarray:
        """
        Evaluate the continuation value estimate Ĉ = X @ β
        given pre-normalised state variables.  Avoids re-normalising
        inside the inner action loop for speed.
        """
        n = len(x_norm)
        cols = [np.ones(n)]
        cols += basis._poly_terms(s_norm)
        if z_norm is not None and model in ("schwartz_smith", "heston"):
            cols += basis._poly_terms(z_norm)
        cols += basis._poly_terms(x_norm)
        if basis.params.cross_terms:
            cols.append(x_norm * s_norm)
            if z_norm is not None and model in ("schwartz_smith", "heston"):
                cols.append(x_norm * z_norm)
        X = np.column_stack(cols)
        return X @ beta

    def _perturbed_greedy_inventory(
        self,
        spots  : np.ndarray,              # (n_fit, n_steps+1)
        model  : str,
        state2 : Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Generate path-consistent inventory trajectories with good
        cross-sectional spread for regression training.

        Method
        ------
        Run a greedy dispatch (inject when next-step price > current, else
        withdraw) and randomly flip ~50% of actions.  This produces
        physically feasible trajectories that span the full inventory range
        at every time step — unlike pure greedy, which converges all paths
        to the same seasonal pattern due to correlated OU prices.

        The 50% flip probability is deliberately high: near the boundaries
        (min/max inventory) infeasible flips are silently kept as the original
        action, so the effective flip rate is somewhat lower in practice.

        Returns
        -------
        inventory : np.ndarray, shape (n_fit, n_steps+1)
        """
        st      = self.storage
        n_fit   = spots.shape[0]
        n_steps = self.n_steps
        inv     = np.empty((n_fit, n_steps + 1))
        inv[:, 0] = st.initial_inventory

        rng = np.random.default_rng(seed=42)

        for t in range(n_steps):
            S_t    = spots[:, t]
            S_next = spots[:, t + 1]
            spread = S_next - S_t   # positive → price rising → inject

            inj_max = self._max_inj[t]
            wit_max = self._max_wit[t]

            # Greedy base action: inject at 50% rate on rising price, withdraw otherwise
            action = np.zeros(n_fit)
            can_inj = (spread > 0) & (inv[:, t] + inj_max * 0.5 <= st.max_inventory)
            action  = np.where(can_inj, inj_max * 0.5, action)
            can_wit = (~can_inj) & (inv[:, t] - wit_max * 0.5 >= st.min_inventory)
            action  = np.where(can_wit, -wit_max * 0.5, action)

            # Random flip for ~50% of paths: inject ↔ withdraw
            flip    = rng.random(n_fit) < 0.5
            flipped = -action
            can_flip_inj = (flipped > 0) & (inv[:, t] + flipped <= st.max_inventory)
            can_flip_wit = (flipped < 0) & (inv[:, t] + flipped >= st.min_inventory)
            can_flip     = can_flip_inj | can_flip_wit | (flipped == 0)
            action       = np.where(flip & can_flip, flipped, action)

            inv[:, t + 1] = np.clip(
                inv[:, t] + action,
                st.min_inventory,
                st.max_inventory,
            )

        return inv
