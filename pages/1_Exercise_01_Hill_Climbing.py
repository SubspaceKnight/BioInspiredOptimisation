"""Exercise 1: Hill Climbing — Theory, Implementation, Interactive Lab & Discussion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from bio_inspired_optimisation.benchmarks import BENCHMARKS
from bio_inspired_optimisation.experiments import run_batch_benchmark, run_single_comparison
from bio_inspired_optimisation.neighborhoods import NEIGHBORHOOD_FACTORIES
from bio_inspired_optimisation.plotting import (
    plot_1d_trajectory,
    plot_2d_contour_trajectory,
    plot_3d_surface_trajectory,
    plot_batch_distribution,
    plot_convergence_and_step_size,
    plot_step_walkthrough_1d,
    plot_step_walkthrough_2d,
)
from bio_inspired_optimisation.termination import TerminationCriterion

st.set_page_config(
    page_title="Exercise 1: Hill Climbing",
    page_icon="⛰️",
    layout="wide",
)

st.title("Exercise 1: Hill Climbing (Minimization)")
st.caption(
    "Derivative-free local search in continuous domains · Simple Hill Climbing vs. "
    "Adam-Inspired Adaptive Momentum Hill Climbing"
)

# ---------------------------------------------------------------------------
# Sidebar: Experiment Configuration (always visible while browsing tabs)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Experiment Configuration")

    bench_key = st.selectbox(
        "Benchmark function",
        options=list(BENCHMARKS.keys()),
        format_func=lambda k: BENCHMARKS[k].name,
        index=2,  # Ackley (2D) default for rich 2D/3D exploration, switchable to 1D
    )
    spec = BENCHMARKS[bench_key]
    dim = spec.default_dim

    algo_mode = st.radio(
        "Algorithm(s) to run",
        options=["Compare Both", "Simple Hill Climbing", "Adaptive Momentum Hill Climbing"],
        index=0,
    )

    with st.expander("Domain & Initial Candidate (x₀)", expanded=True):
        default_lo, default_hi = spec.default_bounds
        if bench_key == "ackley":
            domain_preset = st.selectbox(
                "Ackley zoom / domain preset",
                options=[
                    "Full PDF domain [-10π, 10π]",
                    "Medium [-10, 10]",
                    "Central funnel [-5, 5]",
                    "Custom",
                ],
                index=0,
            )
            if domain_preset == "Full PDF domain [-10π, 10π]":
                lo_val, hi_val = -10.0 * np.pi, 10.0 * np.pi
            elif domain_preset == "Medium [-10, 10]":
                lo_val, hi_val = -10.0, 10.0
            elif domain_preset == "Central funnel [-5, 5]":
                lo_val, hi_val = -5.0, 5.0
            else:
                lo_val = st.number_input("Lower bound", value=float(default_lo), step=1.0)
                hi_val = st.number_input("Upper bound", value=float(default_hi), step=1.0)
        else:
            lo_val = st.number_input("Lower bound", value=float(default_lo), step=0.5)
            hi_val = st.number_input("Upper bound", value=float(default_hi), step=0.5)

        if lo_val >= hi_val:
            st.error("Lower bound must be strictly less than upper bound. Reverting to defaults.")
            lo_val, hi_val = default_lo, default_hi

        bounds = spec.make_bounds(dim=dim, low=lo_val, high=hi_val)

        use_random_x0 = st.checkbox("Randomize x₀ from seed", value=False)
        seed = int(st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1))

        if use_random_x0:
            init_rng = np.random.default_rng(seed)
            x0 = init_rng.uniform(bounds[0], bounds[1])
            st.caption(f"Sampled x₀ = {np.round(x0, 3).tolist()}")
        else:
            if dim == 1:
                default_x0 = 7.5 if bench_key == "quadratic" else -4.2
                default_x0 = float(np.clip(default_x0, lo_val, hi_val))
                x0_val = st.slider(
                    "Initial x₀",
                    min_value=float(lo_val),
                    max_value=float(hi_val),
                    value=default_x0,
                    step=float((hi_val - lo_val) / 200.0),
                )
                x0 = np.array([x0_val], dtype=float)
            else:
                span = hi_val - lo_val
                c1, c2 = st.columns(2)
                x0_1 = c1.slider(
                    "Initial x₁",
                    min_value=float(lo_val),
                    max_value=float(hi_val),
                    value=float(np.clip(-0.35 * hi_val, lo_val, hi_val)),
                    step=float(span / 200.0),
                )
                x0_2 = c2.slider(
                    "Initial x₂",
                    min_value=float(lo_val),
                    max_value=float(hi_val),
                    value=float(np.clip(0.28 * hi_val, lo_val, hi_val)),
                    step=float(span / 200.0),
                )
                x0 = np.array([x0_1, x0_2], dtype=float)

    with st.expander("Neighborhood & Step Size", expanded=True):
        neighborhood_name = st.selectbox(
            "Neighborhood distribution",
            options=list(NEIGHBORHOOD_FACTORIES.keys()),
            index=2 if dim == 2 else 0,
        )
        boundary_mode = st.radio(
            "Boundary handling",
            options=["clip", "reflect"],
            horizontal=True,
            index=0,
        )
        max_reasonable_step = max(0.5, float(hi_val - lo_val) * 0.6)
        default_step = 1.5 if bench_key == "ackley" and (hi_val - lo_val) > 25 else 0.6
        default_step = float(min(default_step, max_reasonable_step))
        step_size = st.slider(
            "Initial step size (σ₀)",
            min_value=0.01,
            max_value=float(round(max_reasonable_step, 2)),
            value=default_step,
            step=0.05,
        )

    with st.expander("Adaptive Momentum (Adam-Inspired)", expanded=False):
        momentum_gain = st.slider(
            "Momentum gain (drift multiplier)",
            min_value=0.0,
            max_value=2.0,
            value=0.75,
            step=0.05,
            help="Set to 0.0 to disable momentum drift and isolate pure adaptive step sizing.",
        )
        beta1 = st.slider("First-moment decay β₁", min_value=0.0, max_value=0.95, value=0.70, step=0.05)
        beta2 = st.slider("Second-moment decay β₂", min_value=0.0, max_value=0.99, value=0.90, step=0.05)
        reject_momentum_decay = st.slider(
            "Momentum damping on reject",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
        )
        c_g, c_s = st.columns(2)
        grow_factor = c_g.number_input("Grow factor", min_value=1.0, max_value=2.5, value=1.25, step=0.05)
        shrink_factor = c_s.number_input("Shrink factor", min_value=0.1, max_value=1.0, value=0.75, step=0.05)
        c_ag, c_rs = st.columns(2)
        accept_streak_to_grow = int(
            c_ag.number_input("Accept streak → grow", min_value=1, max_value=10, value=2, step=1)
        )
        reject_streak_to_shrink = int(
            c_rs.number_input("Reject streak → shrink", min_value=1, max_value=20, value=3, step=1)
        )
        min_step_size = float(
            st.select_slider(
                "Minimum adaptive σ_min",
                options=[1e-8, 1e-6, 1e-4, 1e-3, 1e-2],
                value=1e-6,
            )
        )

    with st.expander("Termination Criteria", expanded=True):
        max_evals = int(
            st.slider("Max objective evaluations", min_value=20, max_value=1500, value=250, step=10)
        )
        use_patience = st.checkbox("Stop after consecutive rejections (patience)", value=False)
        max_no_improve = (
            int(st.slider("Max consecutive rejections", min_value=5, max_value=300, value=50, step=5))
            if use_patience
            else None
        )
        use_min_step_stop = st.checkbox("Stop if adaptive σ ≤ σ_min", value=False)

    termination = TerminationCriterion(
        max_evals=max_evals,
        max_no_improve=max_no_improve,
        min_step_size=min_step_size if use_min_step_stop else None,
    )

# Run single comparison using current sidebar settings
simple_res, adaptive_res = run_single_comparison(
    spec=spec,
    initial_solution=x0,
    bounds=bounds,
    neighborhood_name=neighborhood_name,
    boundary_mode=boundary_mode,  # type: ignore[arg-type]
    step_size=step_size,
    termination=termination,
    seed=seed,
    beta1=beta1,
    beta2=beta2,
    momentum_gain=momentum_gain,
    reject_momentum_decay=reject_momentum_decay,
    grow_factor=float(grow_factor),
    shrink_factor=float(shrink_factor),
    accept_streak_to_grow=accept_streak_to_grow,
    reject_streak_to_shrink=reject_streak_to_shrink,
    min_step_size=min_step_size,
)

if algo_mode == "Simple Hill Climbing":
    active_results = [simple_res]
elif algo_mode == "Adaptive Momentum Hill Climbing":
    active_results = [adaptive_res]
else:
    active_results = [simple_res, adaptive_res]

# ---------------------------------------------------------------------------
# Formatting & Stepping Helpers for Tab 5
# ---------------------------------------------------------------------------
def _fmt_num(val: float, signed: bool = False, decimals: int = 5) -> str:
    if val != 0.0 and abs(val) < 1e-4:
        return f"{val:+.3e}" if signed else f"{val:.3e}"
    return f"{val:+.{decimals}f}" if signed else f"{val:.{decimals}f}"


def _fmt_vec(arr: np.ndarray | None, signed: bool = False, decimals: int = 5) -> str:
    if arr is None:
        return "—"
    flat = np.asarray(arr, dtype=float).ravel()
    if flat.size == 1:
        return _fmt_num(float(flat[0]), signed=signed, decimals=decimals)
    return "[" + ", ".join(_fmt_num(float(v), signed=signed, decimals=decimals) for v in flat) + "]"


def _fmt_latex_vec(
    arr: np.ndarray | None,
    signed: bool = False,
    decimals: int = 5,
    paren_neg: bool = False,
) -> str:
    if arr is None:
        return r"\text{—}"
    flat = np.asarray(arr, dtype=float).ravel()
    if flat.size == 1:
        val = float(flat[0])
        txt = _fmt_num(val, signed=signed, decimals=decimals)
        if paren_neg and val < 0.0:
            return f"({txt})"
        return txt
    inner = r" \\ ".join(_fmt_num(float(v), signed=signed, decimals=decimals) for v in flat)
    return r"\begin{bmatrix} " + inner + r" \end{bmatrix}"


def _walk_step_delta(delta: int, max_eval: int) -> None:
    cur = int(st.session_state.get("walkthrough_step", 1))
    st.session_state["walkthrough_step"] = max(1, min(max_eval, cur + delta))


def _walk_set_step(target: int) -> None:
    st.session_state["walkthrough_step"] = max(1, int(target))


def _close_walk_popup() -> None:
    st.session_state["walk_popup_open"] = False


def _walk_pop_delta(delta: int, max_eval: int) -> None:
    cur = int(st.session_state.get("walk_popup_eval", 1))
    st.session_state["walk_popup_eval"] = max(1, min(max_eval, cur + delta))


def _rng_dist_markdown(
    neighborhood_name: str,
    sigma_in: float,
    dim: int,
    seed_val: int,
    boundary_mode: str,
) -> tuple[str, str]:
    """Return (plain_description, latex_dist) describing how RNG draws offset epsilon."""
    sig_str = _fmt_num(sigma_in, decimals=5)
    dim_sup = f"^{dim}" if dim > 1 else ""
    if "Box" in neighborhood_name:
        plain = (
            f"Sampled pseudo-randomly by RNG (`seed = {seed_val}`) from **Uniform(`-{sig_str}`, `+{sig_str}`){dim_sup}** "
            f"(then bounded via `{boundary_mode}`)"
        )
        lat = rf"\varepsilon \sim \mathcal{{U}}\!\left([-{sig_str},\; +{sig_str}]\right){dim_sup}"
    elif "Gaussian" in neighborhood_name:
        var_str = _fmt_num(sigma_in * sigma_in, decimals=5)
        cov_label = f"{var_str} I_{dim}" if dim > 1 else var_str
        zero_label = r"\vec{0}" if dim > 1 else "0"
        plain = (
            f"Sampled pseudo-randomly by RNG (`seed = {seed_val}`) from **Normal(0, σ²={var_str})** "
            f"(then bounded via `{boundary_mode}`)"
        )
        lat = rf"\varepsilon \sim \mathcal{{N}}\!\left({zero_label},\; {cov_label}\right)"
    else:
        plain = (
            f"Sampled pseudo-randomly by RNG (`seed = {seed_val}`) uniformly inside the **L2 ball ||ε||₂ ≤ `{sig_str}`** "
            f"(then bounded via `{boundary_mode}`)"
        )
        lat = rf"\varepsilon \sim \operatorname{{Uniform}}\!\left(\{{\|\vec{{u}}\|_2 \le {sig_str}\}}\right)"
    return plain, lat


def _build_state_transition_rows(
    rec,
    *,
    is_adaptive: bool,
    reject_momentum_decay: float,
    accept_streak_to_grow: int,
    reject_streak_to_shrink: int,
) -> list[dict[str, str]]:
    sigma_in = float(rec.step_size_before if rec.step_size_before is not None else rec.step_size)
    sigma_out = float(rec.step_size)
    if is_adaptive:
        return [
            {
                "Variable": "Anchor x_curr (best_x)",
                "Before Step": _fmt_vec(rec.prev_x, decimals=5),
                "After Step": _fmt_vec(rec.best_x, decimals=5),
                "Rule": "Update on Accept" if rec.accepted else "Keep previous best anchor",
            },
            {
                "Variable": "Best f(x_best)",
                "Before Step": _fmt_num(rec.prev_f, decimals=5) if rec.prev_f is not None else "—",
                "After Step": _fmt_num(rec.best_f, decimals=5),
                "Rule": "f(x_new) < f_curr" if rec.accepted else "No improvement",
            },
            {
                "Variable": "Step size σ",
                "Before Step": _fmt_num(sigma_in, decimals=5),
                "After Step": _fmt_num(sigma_out, decimals=5),
                "Rule": f"{rec.step_event} (carried to next eval)",
            },
            {
                "Variable": "Momentum steps t_mom",
                "Before Step": str(rec.t_mom_before),
                "After Step": str(rec.t_mom_after),
                "Rule": "+1 on Accept; unchanged on Reject",
            },
            {
                "Variable": "First moment m",
                "Before Step": _fmt_vec(rec.m_before, decimals=5),
                "After Step": _fmt_vec(rec.m_after, decimals=5),
                "Rule": "EMA of δ" if rec.accepted else f"Damp by γ={reject_momentum_decay:.2f}",
            },
            {
                "Variable": "Second moment v",
                "Before Step": _fmt_vec(rec.v_before, decimals=5),
                "After Step": _fmt_vec(rec.v_after, decimals=5),
                "Rule": "EMA of δ²" if rec.accepted else "Unchanged on Reject",
            },
            {
                "Variable": "accept_streak",
                "Before Step": f"{rec.accept_streak_before} / {accept_streak_to_grow}",
                "After Step": f"{rec.accept_streak_after} / {accept_streak_to_grow}",
                "Rule": "Reset at Grow or Reject",
            },
            {
                "Variable": "reject_streak",
                "Before Step": f"{rec.reject_streak_before} / {reject_streak_to_shrink}",
                "After Step": f"{rec.reject_streak_after} / {reject_streak_to_shrink}",
                "Rule": "Reset at Shrink or Accept",
            },
        ]
    return [
        {
            "Variable": "Anchor x_curr (best_x)",
            "Before Step": _fmt_vec(rec.prev_x, decimals=5),
            "After Step": _fmt_vec(rec.best_x, decimals=5),
            "Rule": "Update on Accept" if rec.accepted else "Keep previous best anchor",
        },
        {
            "Variable": "Best f(x_best)",
            "Before Step": _fmt_num(rec.prev_f, decimals=5) if rec.prev_f is not None else "—",
            "After Step": _fmt_num(rec.best_f, decimals=5),
            "Rule": "f(x_new) < f_curr" if rec.accepted else "No improvement",
        },
        {
            "Variable": "Step size σ",
            "Before Step": _fmt_num(sigma_in, decimals=5),
            "After Step": _fmt_num(sigma_out, decimals=5),
            "Rule": "Constant σ₀",
        },
        {
            "Variable": "accept_streak",
            "Before Step": str(rec.accept_streak_before),
            "After Step": str(rec.accept_streak_after),
            "Rule": "Consecutive accepts (informational)",
        },
        {
            "Variable": "reject_streak",
            "Before Step": str(rec.reject_streak_before),
            "After Step": str(rec.reject_streak_after),
            "Rule": "Consecutive rejects (informational)",
        },
    ]


def _render_eval_step_details(
    chosen_walk,
    rec,
    *,
    dim: int,
    bounds: tuple[np.ndarray, np.ndarray],
    lo_val: float,
    hi_val: float,
    neighborhood_name: str,
    boundary_mode: str,
    seed_val: int,
    momentum_gain: float,
    beta1: float,
    beta2: float,
    reject_momentum_decay: float,
    grow_factor: float,
    shrink_factor: float,
    accept_streak_to_grow: int,
    reject_streak_to_shrink: int,
    min_step_size: float,
) -> None:
    """Render full 3-phase step-by-step math and execution order for one evaluation record."""
    is_adaptive_walk = chosen_walk.algorithm_name == "Adaptive Momentum Hill Climbing"
    sigma_in = float(rec.step_size_before if rec.step_size_before is not None else rec.step_size)
    sigma_out = float(rec.step_size)

    if rec.eval_index == 1 or rec.prev_x is None or rec.prev_f is None:
        st.info(
            "**Initialization Step (Evaluation #1):** No neighbor search happens yet — the solver evaluates the starting "
            "point $x_0$, establishes the baseline anchor $(x_{\\text{curr}}, f_{\\text{curr}})$, and initializes all state counters."
        )
        init_lat_x0 = _fmt_latex_vec(rec.x)
        st.latex(
            r"x_{\text{curr}} = x_0 = "
            + init_lat_x0
            + r", \qquad f(x_0) = "
            + _fmt_num(rec.f)
            + r", \qquad \sigma_0 = "
            + _fmt_num(sigma_out, decimals=4)
        )
        if is_adaptive_walk:
            st.latex(
                r"t_{\text{mom}} = 0, \quad m = 0, \quad v = 0, \quad \text{drift} = 0, \quad "
                r"\text{accept\_streak} = 0, \quad \text{reject\_streak} = 0"
            )
            st.markdown(
                "Why is `drift = 0` at the start? No accepted step $\\delta = x_{\\text{new}} - x_{\\text{curr}}$ "
                "has occurred yet ($t_{\\text{mom}} = 0$), so there is no historical descent direction to extrapolate."
            )
        else:
            st.latex(
                r"\text{accept\_streak} = 0, \quad \text{reject\_streak} = 0, \quad "
                r"\sigma = \sigma_0 = " + _fmt_num(sigma_out, decimals=4) + r"\text{ (fixed)}"
            )
        return

    prev_x_arr = rec.prev_x
    prev_f_val = float(rec.prev_f)
    drift_arr = rec.drift if rec.drift is not None else np.zeros_like(rec.x)
    offset_arr = rec.offset if rec.offset is not None else (rec.x - (prev_x_arr + drift_arr))
    center_arr = prev_x_arr + drift_arr
    upper_sigma_val = chosen_walk.params.get("max_step_size")

    st.caption(
        f"**Execution Order for Eval #{rec.eval_index} (Iter {rec.iter_index}):** "
        f"① Carry in best anchor $x_{{\\text{{curr}}}} = \\text{{best\\_x}}$ from Eval #{rec.eval_index - 1} (`{_fmt_vec(prev_x_arr)}`) and $\\sigma_{{\\text{{in}}}} = {_fmt_num(sigma_in, decimals=4)}$ "
        f"→ ② Compute `drift` (Phase 1) "
        f"→ ③ Draw stochastic RNG offset $\\varepsilon$ & evaluate $f(x_{{\\text{{new}}}})$ (Phase 2) "
        f"→ ④ Update anchor, `(m, v)`, streaks & $\\sigma_{{\\text{{after}}}}$ for Eval #{rec.eval_index + 1} (Phase 3)."
    )

    # ---------------- PHASE 1 ----------------
    st.markdown("#### Phase 1 — Before Sampling: Compute `drift` and Search Center")
    if not is_adaptive_walk:
        st.markdown(
            "Simple Hill Climbing has no momentum memory: `drift = 0`, so the search neighborhood is centered "
            f"directly at the current best anchor $x_{{\\text{{curr}}}} = {_fmt_vec(prev_x_arr)}$ with fixed radius $\\sigma_0 = {_fmt_num(sigma_in, decimals=4)}$."
        )
        st.latex(
            r"c = x_{\text{curr}} = "
            + _fmt_latex_vec(prev_x_arr)
            + r", \qquad \text{drift} = 0, \qquad \sigma = "
            + _fmt_num(sigma_in, decimals=4)
        )
    else:
        st.markdown(
            f"Incoming state carried from Eval #{rec.eval_index - 1}: "
            f"$x_{{\\text{{curr}}}} = {_fmt_vec(prev_x_arr)}$ (best so far), "
            f"$t_{{\\text{{mom}}}} = {rec.t_mom_before}$, "
            f"$m_{{\\text{{in}}}} = {_fmt_vec(rec.m_before)}$, "
            f"$v_{{\\text{{in}}}} = {_fmt_vec(rec.v_before)}$, "
            f"$\\sigma_{{\\text{{in}}}} = {_fmt_num(sigma_in, decimals=4)}$, "
            f"`accept_streak` = `{rec.accept_streak_before}/{accept_streak_to_grow}`, "
            f"`reject_streak` = `{rec.reject_streak_before}/{reject_streak_to_shrink}`."
        )
        if rec.t_mom_before == 0:
            st.info(
                "Since **$t_{\\text{mom}} = 0$** (0 accepted moves have occurred prior to this step), bias correction "
                "$1/(1 - \\beta_1^0)$ is undefined and no directional displacement history exists yet. "
                "Therefore **`drift = 0`** and this probe searches isotropically around $x_{\\text{curr}}$."
            )
            st.latex(r"\text{drift} = 0 \implies c = x_{\text{curr}} = " + _fmt_latex_vec(prev_x_arr))
        elif momentum_gain == 0.0:
            st.info(
                "Sidebar **Momentum gain $\\eta_{\\text{mom}} = 0.0$**, so directional bias is disabled (`drift = 0`). "
                "Only streak-based step adaptation of $\\sigma$ remains active."
            )
            st.latex(r"\text{drift} = 0 \cdot \sigma_{\text{in}} \cdot d_{\text{norm}} = 0")
        else:
            denom1 = (1.0 - beta1**rec.t_mom_before) if beta1 > 0.0 else 1.0
            denom2 = (1.0 - beta2**rec.t_mom_before) if beta2 > 0.0 else 1.0
            st.latex(
                r"\hat{m} = \frac{m_{\text{in}}}{1 - \beta_1^{t_{\text{mom}}}} = \frac{"
                + _fmt_latex_vec(rec.m_before)
                + r"}{"
                + f"{denom1:.5f}"
                + r"} = "
                + _fmt_latex_vec(rec.m_hat)
                + r", \qquad \hat{v} = \frac{v_{\text{in}}}{1 - \beta_2^{t_{\text{mom}}}} = \frac{"
                + _fmt_latex_vec(rec.v_before)
                + r"}{"
                + f"{denom2:.5f}"
                + r"} = "
                + _fmt_latex_vec(rec.v_hat)
            )
            st.latex(
                r"d_{\text{norm}} = \operatorname{clip}\!\left(\frac{\hat{m}}{\sqrt{\hat{v}} + \varepsilon_{\text{adam}}}, -1, 1\right) = "
                + _fmt_latex_vec(rec.d_norm, signed=False)
            )
            st.latex(
                r"\text{drift} = \eta_{\text{mom}} \cdot \sigma_{\text{in}} \cdot d_{\text{norm}} = "
                + f"{momentum_gain:.2f} \\cdot {_fmt_num(sigma_in, decimals=4)} \\cdot "
                + _fmt_latex_vec(rec.d_norm, signed=False, paren_neg=True)
                + r" = "
                + _fmt_latex_vec(drift_arr, signed=True)
            )

    # ---------------- PHASE 2 ----------------
    st.markdown("#### Phase 2 — Draw Stochastic Offset $\\varepsilon$ (RNG) & Evaluate Candidate $x_{\\text{new}}$")
    rng_plain, rng_lat = _rng_dist_markdown(
        neighborhood_name=neighborhood_name,
        sigma_in=sigma_in,
        dim=dim,
        seed_val=seed_val,
        boundary_mode=boundary_mode,
    )
    st.markdown(
        f"**Why isn't $\\varepsilon$ computed by a deterministic formula?** In stochastic Hill Climbing, the perturbation $\\varepsilon$ "
        f"is the random exploration step: {rng_plain}, giving `{_fmt_vec(offset_arr, signed=True)}`."
    )
    st.latex(
        rng_lat
        + r" \xrightarrow{\text{RNG}(\text{seed}="
        + f"{seed_val}"
        + r")} \varepsilon = "
        + _fmt_latex_vec(offset_arr, signed=True)
    )
    st.latex(
        r"x_{\text{new}} = \underbrace{"
        + _fmt_latex_vec(prev_x_arr)
        + r"}_{x_{\text{curr}}} + \underbrace{"
        + _fmt_latex_vec(drift_arr, signed=True, paren_neg=True)
        + r"}_{\text{drift}} + \underbrace{"
        + _fmt_latex_vec(offset_arr, signed=True, paren_neg=True)
        + r"}_{\varepsilon} = "
        + _fmt_latex_vec(rec.x)
    )
    if np.any(center_arr < bounds[0]) or np.any(center_arr > bounds[1]):
        st.caption(
            f"Note: Shifted center {_fmt_vec(center_arr)} extended past domain bounds [{lo_val:.4f}, {hi_val:.4f}] "
            f"and was mapped back via `{boundary_mode}`; displayed ε is the net post-boundary displacement from $x_{{\\text{{curr}}}} + \\text{{drift}}$."
        )

    diff_f = rec.f - prev_f_val
    cmp_symbol = "<" if rec.accepted else r"\ge"
    decision_word = r"\textbf{ACCEPT} \text{ (new minimum)}" if rec.accepted else r"\textbf{REJECT} \text{ (keep old anchor)}"
    st.latex(
        r"f(x_{\text{new}}) = "
        + _fmt_num(rec.f)
        + r"\quad "
        + cmp_symbol
        + r"\quad f(x_{\text{curr}}) = "
        + _fmt_num(prev_f_val)
        + r"\qquad (\Delta f = "
        + _fmt_num(diff_f, signed=True)
        + r") \implies "
        + decision_word
    )

    # ---------------- PHASE 3 ----------------
    st.markdown("#### Phase 3 — Update Anchor, Moments `(m, v)`, Streaks, and Step Size $\\sigma$")
    if rec.accepted:
        st.success(
            f"**Accepted ({rec.step_event}):** Candidate becomes the new anchor (`best_x = {_fmt_vec(rec.best_x)}`, `best_f = {_fmt_num(rec.best_f)}`) carried into Eval #{rec.eval_index + 1}."
        )
        delta_v = rec.delta if rec.delta is not None else (rec.x - prev_x_arr)
        if is_adaptive_walk:
            st.latex(
                r"\delta = x_{\text{new}} - x_{\text{prev}} = "
                + _fmt_latex_vec(rec.x)
                + r" - "
                + _fmt_latex_vec(prev_x_arr, paren_neg=True)
                + r" = "
                + _fmt_latex_vec(delta_v, signed=True)
            )
            st.latex(
                r"t_{\text{mom}}: "
                + f"{rec.t_mom_before} \\to {rec.t_mom_after}, \\qquad "
                + r"m_{\text{after}} = \beta_1 m_{\text{in}} + (1 - \beta_1)\delta = "
                + f"{beta1:.2f} \\cdot "
                + _fmt_latex_vec(rec.m_before, paren_neg=True)
                + f" + {1.0 - beta1:.2f} \\cdot "
                + _fmt_latex_vec(delta_v, signed=False, paren_neg=True)
                + r" = "
                + _fmt_latex_vec(rec.m_after)
            )
            delta_sq = delta_v**2
            st.latex(
                r"v_{\text{after}} = \beta_2 v_{\text{in}} + (1 - \beta_2)\delta^2 = "
                + f"{beta2:.2f} \\cdot "
                + _fmt_latex_vec(rec.v_before)
                + f" + {1.0 - beta2:.2f} \\cdot "
                + _fmt_latex_vec(delta_sq)
                + r" = "
                + _fmt_latex_vec(rec.v_after)
            )

            acc_reached = rec.accept_streak_before + 1
            if rec.step_event == "Accept (Grow σ)":
                cap_str = (
                    f"\\min({_fmt_num(float(upper_sigma_val), decimals=4)}, \\; {_fmt_num(sigma_in * float(grow_factor), decimals=5)})"
                    if upper_sigma_val is not None
                    else _fmt_num(sigma_in * float(grow_factor), decimals=5)
                )
                st.latex(
                    r"\text{accept\_streak}: "
                    + f"{rec.accept_streak_before} + 1 = {acc_reached} \\ge {accept_streak_to_grow}"
                    + r" \implies \sigma_{\text{after}} = "
                    + cap_str
                    + r" = "
                    + _fmt_num(sigma_out, decimals=5)
                    + r"\text{ (then } \text{accept\_streak} \leftarrow 0\text{, } \text{reject\_streak} \leftarrow 0\text{)}"
                )
            else:
                st.latex(
                    r"\text{accept\_streak}: "
                    + f"{rec.accept_streak_before} \\to {rec.accept_streak_after} < {accept_streak_to_grow}"
                    + r", \quad \text{reject\_streak}: "
                    + f"{rec.reject_streak_before} \\to 0"
                    + r" \implies \sigma_{\text{after}} = "
                    + _fmt_num(sigma_out, decimals=5)
                    + r"\text{ (unchanged)}"
                )
        else:
            st.latex(
                r"\text{accept\_streak}: "
                + f"{rec.accept_streak_before} \\to {rec.accept_streak_after}, \\quad "
                + r"\text{reject\_streak}: "
                + f"{rec.reject_streak_before} \\to 0, \\quad "
                + r"\sigma_{\text{after}} = \sigma_0 = "
                + _fmt_num(sigma_out, decimals=4)
                + r"\text{ (fixed)}"
            )
    else:
        st.error(
            f"**Rejected ({rec.step_event}):** Candidate discarded; Eval #{rec.eval_index + 1} stays anchored at the best point `best_x = {_fmt_vec(rec.best_x)}`, `best_f = {_fmt_num(rec.best_f)}`."
        )
        if is_adaptive_walk:
            st.latex(
                r"t_{\text{mom}} = "
                + f"{rec.t_mom_after}\\text{{ (unchanged)}}, \\qquad "
                + r"v_{\text{after}} = v_{\text{in}} = "
                + _fmt_latex_vec(rec.v_after)
            )
            st.latex(
                r"m_{\text{after}} = \gamma_{\text{reject}} \cdot m_{\text{in}} = "
                + f"{reject_momentum_decay:.2f} \\cdot "
                + _fmt_latex_vec(rec.m_before, paren_neg=True)
                + r" = "
                + _fmt_latex_vec(rec.m_after)
            )
            rej_reached = rec.reject_streak_before + 1
            if rec.step_event == "Reject (Shrink σ)":
                st.latex(
                    r"\text{reject\_streak}: "
                    + f"{rec.reject_streak_before} + 1 = {rej_reached} \\ge {reject_streak_to_shrink}"
                    + r" \implies \sigma_{\text{after}} = \max("
                    + f"{min_step_size:.1e},\\; {_fmt_num(sigma_in, decimals=4)} \\times {shrink_factor:.2f}) = "
                    + _fmt_num(sigma_out, decimals=5)
                    + r"\text{ (then } \text{reject\_streak} \leftarrow 0\text{, } \text{accept\_streak} \leftarrow 0\text{)}"
                )
            else:
                st.latex(
                    r"\text{reject\_streak}: "
                    + f"{rec.reject_streak_before} \\to {rec.reject_streak_after} < {reject_streak_to_shrink}"
                    + r", \quad \text{accept\_streak}: "
                    + f"{rec.accept_streak_before} \\to 0"
                    + r" \implies \sigma_{\text{after}} = "
                    + _fmt_num(sigma_out, decimals=5)
                    + r"\text{ (unchanged)}"
                )
        else:
            st.latex(
                r"\text{reject\_streak}: "
                + f"{rec.reject_streak_before} \\to {rec.reject_streak_after}, \\quad "
                + r"\text{accept\_streak}: "
                + f"{rec.accept_streak_before} \\to 0, \\quad "
                + r"\sigma_{\text{after}} = \sigma_0 = "
                + _fmt_num(sigma_out, decimals=4)
                + r"\text{ (fixed)}"
            )


# ---------------------------------------------------------------------------
# Main Page Sections (PDF 1.3 required structure + Interactive Walkthrough)
# ---------------------------------------------------------------------------
tab_intro, tab_methods, tab_results, tab_discussion, tab_walkthrough = st.tabs(
    [
        "1. Introduction",
        "2. Methods",
        "3. Results",
        "4. Discussion",
        "5. Step-by-Step Dynamics",
    ]
)

# ============================ 1. INTRODUCTION ==============================
with tab_intro:
    st.header("1. Introduction")
    st.markdown(
        """
**Hill Climbing** is one of the simplest single-solution **local search** metaheuristics in
bio-inspired and black-box optimization. Starting from a single initial candidate
$\\vec{x}_0 \\in \\mathbb{R}^n$, the algorithm repeatedly probes a nearby candidate
$\\vec{x}_{\\text{new}} \\in N(\\vec{x}_{\\text{old}})$ and moves only when the candidate strictly improves
the objective value.

### Minimization vs. Maximization
Historically, the name *"Hill Climbing"* comes from maximizing a fitness or reward surface
($f(\\vec{x}_{\\text{new}}) > f(\\vec{x}_{\\text{old}})$). For benchmark **minimization** tasks
(finding valleys rather than peaks), the exact same mechanism applies with a reversed comparison:
$$f(\\vec{x}_{\\text{new}}) < f(\\vec{x}_{\\text{old}})$$

### Zero-Order (Hill Climbing) vs. Gradient-Based Methods (Adam / GD)
Unlike gradient-based optimization methods commonly used in machine learning, standard
Hill Climbing does not compute derivatives $\\nabla f(\\vec{x})$:

| Property | Hill Climbing (Zero-Order) | Gradient Descent / Adam (First-Order) | Newton's Method (Second-Order) |
| :--- | :--- | :--- | :--- |
| **Information queried** | Function values $f(\\vec{x})$ only (black-box) | Values + gradient $\\nabla f(\\vec{x})$ | Gradient + Hessian $\\nabla^2 f(\\vec{x})$ |
| **How next step is chosen** | Sample neighbor $\\vec{x} + \\Delta\\vec{x}$ and test $f$ | Step along $-\\alpha\\,\\nabla f(\\vec{x})$ (with moments in Adam) | Solve $\\vec{x}_{t+1} = \\vec{x}_t - H^{-1}\\nabla f(\\vec{x}_t)$ |
| **Can jump straight to $\\nabla f = 0$?** | No (gradient is unknown) | No (nonlinear landscapes have many local roots & saddles) | Only locally; $O(n^3)$ cost and can jump to maxima |
| **Works on non-differentiable / noisy $f$?** | **Yes** | Not without surrogates / smoothing | No |

### Pseudocode: Simple Hill Climbing (Minimization)
```text
input: initial_solution x0, objective_fn f, neighborhood_fn N, termination
x_curr <- x0
f_curr <- f(x_curr)        # eval_count = 1
while not termination.reached():
    x_new <- N(x_curr)
    f_new <- f(x_new)      # eval_count += 1
    if f_new < f_curr:
        x_curr <- x_new
        f_curr <- f_new
return x_curr, f_curr
```

### Complexity, Strengths, and Weaknesses
- **Space complexity:** $O(n)$ — stores only the current best candidate vector (plus optional small moment vectors $O(n)$ in our adaptive variant).
- **Time complexity:** $O(E \\cdot (n + C_f))$ where $E$ is the number of objective evaluations and $C_f$ is the cost of evaluating $f(\\vec{x})$. Because $C_f$ dominates in practice, **evaluation count** is the primary runtime and efficiency metric.
- **Strengths:** Simple to implement, $O(n)$ memory footprint, requires no derivatives or differentiability assumptions, and achieves monotonic improvement within a single unimodal basin.
- **Weaknesses:** Purely greedy (never accepts worse solutions) $\\implies$ **gets trapped in local minima**, stalls on **flat plateaus**, struggles along **narrow curved ridges**, and fixed step sizes trade off global reach against local precision.
"""
    )

# ============================== 2. METHODS =================================
with tab_methods:
    st.header("2. Methods")
    st.markdown(
        """
### Reusable Software Architecture
To allow reuse in upcoming exercises, the optimization core is decoupled from the Streamlit UI inside
`src/bio_inspired_optimisation/`:
- **`benchmarks.py`:** Implements $n$-dimensional benchmark objectives (`quadratic`, `sinusoidal`, `ackley`) with domain specs.
- **`neighborhoods.py`:** Continuous perturbation factories (`Uniform Box`, `Uniform L2 Ball`, `Gaussian`) with configurable boundary handling (`clip` or mirror `reflect`).
- **`termination.py`:** Unified `TerminationCriterion` supporting `max_evals`, `max_iters`, `max_no_improve` (patience), `min_step_size`, and `target_fitness`.
- **`hill_climbing.py`:** Solver functions taking `(initial_solution, objective_fn, neighborhood_fn, termination, ...)` and logging every single probe through `TrackedObjective` so `eval_count` is exact.

### Benchmark Functions Implemented
1. **Quadratic (1D):** $f(x) = x^2$ on $[-10, 10]$, global minimum $x^* = 0, f(x^*) = 0$.
2. **Sinusoidal (1D):** $f(x) = \\sin(x)$ on $[-2\\pi, 2\\pi]$, minima at $x^* = -\\pi/2 + 2\\pi k, f(x^*) = -1$.
3. **Ackley (2D):**
   $$f(\\vec{x}) = -20\\exp\\!\\left(-0.2\\sqrt{\\frac{1}{n}\\sum_{i=1}^n x_i^2}\\right) - \\exp\\!\\left(\\frac{1}{n}\\sum_{i=1}^n \\cos(2\\pi x_i)\\right) + 20 + e$$
   evaluated on $[-10\\pi, 10\\pi]^2$ (correcting the missing minus sign on the lower bound in the assignment footnote), global minimum $\\vec{x}^* = (0, 0), f(\\vec{x}^*) = 0$.

### Algorithms & Critical Design Choices

#### 1. Simple Hill Climbing (`simple_hill_climbing`)
Because a continuous neighborhood $B_\\sigma(\\vec{x}) \\subset \\mathbb{R}^n$ contains infinitely many points, Simple Hill Climbing samples **one** neighbor per iteration around $\\vec{x}_{\\text{curr}}$ with fixed radius $\\sigma_0$ and accepts immediately if $f(\\vec{x}_{\\text{new}}) < f(\\vec{x}_{\\text{curr}})$.
- **Pros:** Minimal evaluations per iteration (1), unbiased isotropic exploration, easy to tune ($\\sigma_0$ only).
- **Cons:** Fixed $\\sigma_0$ cannot simultaneously traverse wide domains fast *and* refine minima to high precision; wastes many evaluations bouncing around once $|\\vec{x} - \\vec{x}^*| < \\sigma_0$.

#### 2. Adaptive Momentum Hill Climbing (`adaptive_momentum_hill_climbing`)
Analogous to **Adam** in maintaining first and second moment estimates, this variant remains strictly derivative-free and relies solely on function evaluations. Instead of gradients, it tracks accepted step displacements $\\vec{\\delta}_t = \\vec{x}_{\\text{new}} - \\vec{x}_{\\text{curr}}$:
- **First & Second Moments on Accepted Steps:**
  $$\\vec{m}_t = \\beta_1\\vec{m}_{t-1} + (1-\\beta_1)\\vec{\\delta}_t, \\qquad \\vec{v}_t = \\beta_2\\vec{v}_{t-1} + (1-\\beta_2)\\vec{\\delta}_t^2$$
  with bias corrections $\\hat{\\vec{m}}_t = \\vec{m}_t / (1-\\beta_1^t)$ and $\\hat{\\vec{v}}_t = \\vec{v}_t / (1-\\beta_2^t)$.
- **Momentum-Biased Proposal Center:**
  $$\\vec{\\text{drift}}_t = \\eta_{\\text{mom}}\\cdot \\sigma_t \\cdot \\operatorname{clip}\\!\\left(\\frac{\\hat{\\vec{m}}_t}{\\sqrt{\\hat{\\vec{v}}_t} + \\varepsilon}, -1, 1\\right), \\qquad \\vec{x}_{\\text{new}} = \\vec{x}_{\\text{curr}} + \\vec{\\text{drift}}_t + \\vec{\\epsilon}_t, \\quad \\vec{\\epsilon}_t \\sim N_{\\sigma_t}(\\vec{0})$$
  Scaling drift by $\\sigma_t$ ensures directional momentum shrinks proportionally as the search zooms into a minimum. On rejected probes, $\\vec{m}$ is damped ($\\vec{m} \\leftarrow \\gamma_{\\text{reject}}\\vec{m}$) so the search does not keep pushing into uphill barriers.
- **Streak-Based Step Adaptation ($\\sigma_t$):** Consecutive accepted moves expand $\\sigma_t \\leftarrow \\sigma_t \\cdot c_{\\text{grow}}$ to cross slopes/plateaus faster; consecutive rejections shrink $\\sigma_t \\leftarrow \\max(\\sigma_{\\min}, \\sigma_t \\cdot c_{\\text{shrink}})$ to zoom into fine minima.
- **Pros:** Faster progress along consistent slopes due to directional bias and higher local precision near smooth optima via step-size decay.
- **Cons:** More hyperparameters; if $\\sigma_t$ shrinks aggressively in a multimodal ripple (like outer Ackley), it can lock into a local trap earlier than a large fixed $\\sigma_0$.

### Fair Comparison Protocol
- Both variants start from the exact same $\\vec{x}_0$, use the same RNG seed, neighborhood family, and domain bounds.
- Every call to `objective_fn` (including initial $f(\\vec{x}_0)$) increments `eval_count` via `TrackedObjective`.
- All convergence plots use **objective function evaluations** on the x-axis.
"""
    )

# ============================== 3. RESULTS =================================
with tab_results:
    st.header("3. Results — Interactive Lab & Fog of War")

    max_recorded_eval = max(r.eval_count for r in active_results)

    ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1.2, 1.3, 1.1, 1.1, 1.3])
    fog_of_war = ctrl1.toggle(
        "Fog of War (Black-Box)",
        value=False,
        help="Hide the underlying analytical landscape and display only the evaluated probes and current neighborhood.",
    )
    show_rejected = ctrl2.checkbox("Show rejected probes (×)", value=True)
    show_neighborhood = ctrl3.checkbox("Show neighborhood (±σ)", value=True)
    log_gap = ctrl4.checkbox("Log scale |f - f*|", value=(bench_key in ("quadratic", "ackley")))
    if max_recorded_eval > 1:
        replay_step = ctrl5.slider(
            "Evaluation replay step",
            min_value=1,
            max_value=max_recorded_eval,
            value=max_recorded_eval,
            step=1,
            help="Scrub evaluation-by-evaluation to observe exploration and local-minimum trapping.",
        )
    else:
        replay_step = 1
        ctrl5.caption("Evaluation replay: 1 / 1")

    if fog_of_war:
        st.warning(
            "**Fog of War active:** The underlying analytical landscape is hidden — only the points "
            "evaluated by the optimizer up to step "
            f"**{replay_step}/{max_recorded_eval}** are shown. Disable Fog of War to reveal the full surface."
        )

    # Trajectory Visualizations
    if dim == 1:
        fig_1d = plot_1d_trajectory(
            spec=spec,
            results=active_results,
            bounds_x=(lo_val, hi_val),
            fog_of_war=fog_of_war,
            max_eval_step=replay_step,
            show_rejected=show_rejected,
            show_neighborhood=show_neighborhood,
        )
        st.plotly_chart(fig_1d, width="stretch", theme=None if fog_of_war else "streamlit")
    else:
        fig_2d = plot_2d_contour_trajectory(
            spec=spec,
            results=active_results,
            bounds_x=(lo_val, hi_val),
            bounds_y=(lo_val, hi_val),
            fog_of_war=fog_of_war,
            max_eval_step=replay_step,
            show_rejected=show_rejected,
            show_neighborhood=show_neighborhood,
        )
        fig_3d = plot_3d_surface_trajectory(
            spec=spec,
            results=active_results,
            bounds_x=(lo_val, hi_val),
            bounds_y=(lo_val, hi_val),
            fog_of_war=fog_of_war,
            max_eval_step=replay_step,
            show_rejected=show_rejected,
        )
        layout_2d_3d = st.radio(
            "2D / 3D layout",
            options=["Full-Width Stacked", "Side-by-Side"],
            horizontal=True,
            index=0,
        )
        if layout_2d_3d == "Side-by-Side":
            view_c1, view_c2 = st.columns(2)
            with view_c1:
                st.plotly_chart(fig_2d, width="stretch", theme=None if fog_of_war else "streamlit")
            with view_c2:
                st.plotly_chart(fig_3d, width="stretch", theme=None if fog_of_war else "streamlit")
        else:
            st.plotly_chart(fig_2d, width="stretch", theme=None if fog_of_war else "streamlit")
            st.plotly_chart(fig_3d, width="stretch", theme=None if fog_of_war else "streamlit")

    # Convergence + Adaptive Step Size Plot
    fig_conv = plot_convergence_and_step_size(
        results=active_results,
        global_min_f=spec.global_min_f,
        max_eval_step=replay_step,
        log_gap=log_gap,
    )
    st.plotly_chart(fig_conv, width="stretch")

    # Run Summary Metrics Table
    st.subheader("Single-Run Summary (at Replay Step)")
    summary_rows = []
    for res in active_results:
        recs = [r for r in res.history if r.eval_index <= replay_step]
        last_rec = recs[-1]
        accepted_so_far = sum(1 for r in recs if r.accepted)
        evals_so_far = len(recs)
        gap = abs(last_rec.best_f - spec.global_min_f)
        summary_rows.append(
            {
                "Algorithm": res.algorithm_name,
                "Initial f(x₀)": round(res.initial_f, 6),
                "Best f(x)": round(last_rec.best_f, 8),
                "Gap |f - f*|": f"{gap:.3e}",
                "Best x": np.round(last_rec.best_x, 5).tolist(),
                "Current σ": f"{last_rec.step_size:.3e}",
                "Evals": evals_so_far,
                "Accepted": accepted_so_far,
                "Accept Rate": f"{100.0 * accepted_so_far / max(1, evals_so_far):.1f}%",
                "Stop Reason (full run)": res.stop_reason,
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")

    with st.expander("Inspect Evaluation Probe Log"):
        selected_log_algo = st.selectbox(
            "Algorithm log",
            options=[r.algorithm_name for r in active_results],
            key="log_algo_select",
        )
        chosen_res = next(r for r in active_results if r.algorithm_name == selected_log_algo)
        log_rows = [
            {
                "eval": rec.eval_index,
                "iter": rec.iter_index,
                "x": np.round(rec.x, 5).tolist(),
                "f(x)": rec.f,
                "accepted": rec.accepted,
                "best_f": rec.best_f,
                "step_size_σ": rec.step_size,
                "drift": np.round(rec.drift, 5).tolist() if rec.drift is not None else None,
            }
            for rec in chosen_res.history
            if rec.eval_index <= replay_step
        ]
        st.dataframe(pd.DataFrame(log_rows), hide_index=True, width="stretch")

    st.divider()

    # Multi-start Paired Statistical Benchmark
    st.subheader("Paired Multi-Start Comparison (Random Initial Points)")
    st.caption(
        "Runs both variants from the exact same N random starting coordinates within the current "
        "domain to measure robustness, local-minimum trapping, and evaluation efficiency."
    )
    b1, b2, b3 = st.columns(3)
    n_runs = int(b1.slider("Number of paired random starts", min_value=5, max_value=50, value=20, step=5))
    success_tol = float(
        b2.select_slider(
            "Success tolerance |best_f - f*| ≤",
            options=[1e-4, 1e-2, 1e-1, 0.5, 1.0],
            value=1e-2,
        )
    )
    batch_seed = int(b3.number_input("Batch seed", min_value=0, max_value=99999, value=123, step=1))

    runs_df, agg_df = run_batch_benchmark(
        spec=spec,
        dim=dim,
        bounds=bounds,
        n_runs=n_runs,
        neighborhood_name=neighborhood_name,
        boundary_mode=boundary_mode,  # type: ignore[arg-type]
        step_size=step_size,
        termination=termination,
        base_seed=batch_seed,
        success_tol=success_tol,
        adaptive_kwargs={
            "beta1": beta1,
            "beta2": beta2,
            "momentum_gain": momentum_gain,
            "reject_momentum_decay": reject_momentum_decay,
            "grow_factor": float(grow_factor),
            "shrink_factor": float(shrink_factor),
            "accept_streak_to_grow": accept_streak_to_grow,
            "reject_streak_to_shrink": reject_streak_to_shrink,
            "min_step_size": min_step_size,
        },
    )

    st.dataframe(
        agg_df.style.format(
            {
                "success_rate_pct": "{:.1f}%",
                "mean_best_f": "{:.5f}",
                "median_best_f": "{:.5f}",
                "std_best_f": "{:.5f}",
                "min_best_f": "{:.6f}",
                "max_best_f": "{:.5f}",
                "mean_evals": "{:.1f}",
                "mean_accept_rate_pct": "{:.1f}%",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    fig_batch = plot_batch_distribution(runs_df, spec.global_min_f)
    st.plotly_chart(fig_batch, width="stretch")

# ============================ 4. DISCUSSION ================================
with tab_discussion:
    st.header("4. Discussion")

    st.markdown(
        """
### Expected vs. Unexpected Findings

1. **Quadratic 1D ($f(x) = x^2$) — Unimodal Precision vs. Resolution Floor**
   - **Expected:** Both Simple Hill Climbing and Adaptive Momentum Hill Climbing reliably find the basin around $x^* = 0$ because the landscape has no local minima.
   - **Key contrast:** Fixed-step Simple HC descends roughly linearly and then hits a **step-size floor** — once $|x| < \\sigma_0$, most random probes overshoot $0$ or land worse, stalling precision around $O(\\sigma_0^2)$. Adaptive Momentum HC accelerates downhill via momentum drift $\\vec{m}_t$ and step growth, then rapidly shrinks $\\sigma_t$ upon overshooting to achieve several orders of magnitude finer precision (down toward $\\sigma_{\\min}$) within the same evaluation budget.

2. **Sinusoidal 1D ($f(x) = \\sin(x)$) — Basin Selection & Boundary Traps**
   - **Expected:** Starting near $-\\pi/2$ or $+3\\pi/2$ converges cleanly to $f^* = -1$. Starting outside the crests ($x_0 < -3\\pi/2$ or $x_0 > \\pi/2$ when bounded to $[-2\\pi, 2\\pi]$) can push Simple HC toward a domain **boundary trap** ($x = \\pm 2\\pi, f = 0$) if $\\sigma_0$ is smaller than the barrier width.
   - **Fog of War insight:** Enabling Fog of War illustrates why: from inside a basin, every rejected probe outside the current downhill side returns a higher value, providing no signal that a deeper valley lies beyond the ridge.

3. **Ackley 2D — Flat Outer Plateau & Local Cosine Ripples (Trade-off)**
   - **Expected:** On $[-10\\pi, 10\\pi]^2$, starting far from the origin places both algorithms on Ackley's nearly flat outer plateau covered in periodic cosine ripples ($\\cos(c x_i)$ has spatial period $1$ when $c = 2\\pi$).
   - **Unexpected / Notable Finding:** **Adaptive step-size shrinking introduces a trade-off on multimodal functions.**
     - *Advantage:* Inside the central funnel $[-5, 5]^2$, Adaptive Momentum HC converges toward $(0, 0)$ with substantially higher final precision than fixed-step Simple HC.
     - *Pitfall:* When starting far out on Ackley with small/medium $\\sigma_0$ and encountering consecutive rejected probes inside a shallow outer cosine ripple, shrinking $\\sigma_t$ prematurely traps Adaptive HC inside that local crater — whereas Simple HC with a constant $\\sigma_0 > 1.0$ retains enough step reach to occasionally step across adjacent cosine basins. This explains why tuning `momentum_gain`, `reject_streak_to_shrink`, and `grow_factor` noticeably affects Ackley success rates.

### Parameter Dependency Summary
| Parameter | Effect when too small | Effect when too large |
| :--- | :--- | :--- |
| **Initial step size $\\sigma_0$** | Slow progress across slopes; easily trapped in tiny local ripples | Skips over narrow basins/minima; low acceptance rate near optimum |
| **Initial point $\\vec{x}_0$** | Determines the basin of attraction completely for single-trajectory HC | — |
| **Momentum gain / $\\beta_1$** | No directional memory; slower along smooth corridors | Can overshoot curved valleys or push repeatedly into barriers before damping |
| **Shrink rate / patience** | Fails to refine final decimal digits within `max_evals` | Collapses $\\sigma_t$ prematurely inside outer local minima on Ackley |

### Limitations & Possible Improvements
1. **No mechanism to escape local optima:** Strictly monotonic acceptance ($f_{\\text{new}} < f_{\\text{curr}}$) guarantees trapping in the first local minimum whose barrier exceeds $\\sigma$.
   - *Possible extension:* **Random-Restart Hill Climbing** (restart from multiple random $\\vec{x}_0$ within the evaluation budget) or **Simulated Annealing** (accept worse moves with probability $e^{-\\Delta f / T}$).
2. **Isotropic sampling ignores ridge curvature:** In higher dimensions or elongated valleys, isotropic perturbations frequently step uphill across steep valley walls.
   - *Possible extension:* Covariance adaptation (**CMA-ES** or **(1+1)-ES with covariance**) or **Tabu Search** memory to avoid revisiting rejected regions.
3. **Budget-aware exploration schedule:** Keeping a larger exploration $\\sigma$ early (or combining global Latin Hypercube or Sobol initial sampling with Adaptive Momentum local refinement) can substantially improve Ackley global success rate.
"""
    )

# ======================= 5. STEP-BY-STEP DYNAMICS ==========================
with tab_walkthrough:
    st.header("5. Step-by-Step Dynamics — Interactive Equations & Neighborhood Walkthrough")
    st.caption(
        "Inspect one objective evaluation at a time using the initial parameters from the left sidebar "
        "(Experiment Configuration). Step forward/backward to observe how anchor, drift, random offset ε, "
        "moments (m, v), streaks, and step size σ evolve dynamically."
    )

    walk_algo_map = {r.algorithm_name: r for r in active_results}
    walk_algo_names = list(walk_algo_map.keys())
    preferred_default_algo = (
        "Adaptive Momentum Hill Climbing"
        if "Adaptive Momentum Hill Climbing" in walk_algo_map
        else walk_algo_names[0]
    )
    if st.session_state.get("walkthrough_algo") not in walk_algo_map:
        st.session_state["walkthrough_algo"] = preferred_default_algo

    ctrl_a, ctrl_b, ctrl_c = st.columns([2.2, 1.4, 1.4])
    walk_algo_name = ctrl_a.selectbox(
        "Algorithm to inspect",
        options=walk_algo_names,
        key="walkthrough_algo",
    )
    walk_zoom = ctrl_b.toggle(
        "Zoom into search window",
        value=False,
        key="walkthrough_zoom",
        help="Center and zoom the plot around the active anchor, drift shift, ±σ window, and candidate x_new.",
    )
    walk_fog = ctrl_c.toggle(
        "Fog of War (Black-Box)",
        value=False,
        key="walkthrough_fog",
        help="Hide the analytical function landscape so only evaluated points and the current neighborhood appear.",
    )

    chosen_walk = walk_algo_map[walk_algo_name]
    is_adaptive_walk = chosen_walk.algorithm_name == "Adaptive Momentum Hill Climbing"
    max_eval_w = chosen_walk.eval_count

    if "walkthrough_step" not in st.session_state:
        st.session_state["walkthrough_step"] = min(4, max_eval_w) if max_eval_w >= 4 else max_eval_w
    st.session_state["walkthrough_step"] = max(1, min(max_eval_w, int(st.session_state["walkthrough_step"])))

    # Quick-jump event discovery (first Grow / first Shrink / first Accept)
    first_grow_eval = next(
        (r.eval_index for r in chosen_walk.history if r.step_event == "Accept (Grow σ)"),
        None,
    )
    first_shrink_eval = next(
        (r.eval_index for r in chosen_walk.history if r.step_event == "Reject (Shrink σ)"),
        None,
    )

    nav_c1, nav_c2, nav_c3, nav_c4, nav_c5 = st.columns([0.9, 3.2, 0.9, 1.1, 1.1])
    cur_w_step = int(st.session_state["walkthrough_step"])
    nav_c1.button(
        "◀ Prev",
        key="walk_prev",
        width="stretch",
        disabled=(cur_w_step <= 1),
        on_click=_walk_step_delta,
        args=(-1, max_eval_w),
    )
    if max_eval_w > 1:
        walk_step = int(
            nav_c2.slider(
                "Inspect Evaluation #",
                min_value=1,
                max_value=max_eval_w,
                step=1,
                key="walkthrough_step",
            )
        )
    else:
        walk_step = 1
        nav_c2.caption("Only 1 evaluation recorded (stopped at Init).")
    nav_c3.button(
        "Next ▶",
        key="walk_next",
        width="stretch",
        disabled=(walk_step >= max_eval_w),
        on_click=_walk_step_delta,
        args=(1, max_eval_w),
    )
    nav_c4.button(
        "1st Grow" if first_grow_eval is None else f"Grow (@{first_grow_eval})",
        key="walk_first_grow",
        width="stretch",
        disabled=(first_grow_eval is None),
        on_click=_walk_set_step,
        args=(first_grow_eval if first_grow_eval is not None else 1,),
    )
    nav_c5.button(
        "1st Shrink" if first_shrink_eval is None else f"Shrink (@{first_shrink_eval})",
        key="walk_first_shrink",
        width="stretch",
        disabled=(first_shrink_eval is None),
        on_click=_walk_set_step,
        args=(first_shrink_eval if first_shrink_eval is not None else 1,),
    )

    if walk_fog:
        st.info(
            f"**Fog of War active in Step Walkthrough:** Analytical landscape hidden up to Eval **{walk_step}/{max_eval_w}**."
        )

    with st.expander("Active Sidebar Parameters (Static Initial Configuration)", expanded=True):
        p_col1, p_col2 = st.columns(2)
        p_col1.markdown(
            f"""
- **Benchmark:** `{spec.name}` ($n = {dim}$, global $f^* = {spec.global_min_f:g}$)
- **Domain Bounds:** `[{lo_val:.4f}, {hi_val:.4f}]` (mode: `{boundary_mode}`)
- **Initial Candidate $x_0$:** `{_fmt_vec(chosen_walk.initial_x)}` ($f(x_0) = {chosen_walk.initial_f:.5f}$)
- **Initial Step Size $\\sigma_0$:** `{step_size:.4g}`
- **Neighborhood & Seed:** `{neighborhood_name}` · `seed = {seed}`
"""
        )
        if is_adaptive_walk:
            upper_sigma_val = chosen_walk.params.get("max_step_size")
            upper_sigma_str = f"{float(upper_sigma_val):.4g}" if upper_sigma_val is not None else "None"
            eps_val = float(chosen_walk.params.get("eps", 1e-8))
            p_col2.markdown(
                f"""
- **Momentum Gain $\\eta_{{\\text{{mom}}}}$:** `{momentum_gain:.2f}` · **Reject Damping $\\gamma_{{\\text{{reject}}}}$:** `{reject_momentum_decay:.2f}`
- **EMA Decays:** $\\beta_1 = {beta1:.2f}$, $\\beta_2 = {beta2:.2f}$, $\\varepsilon_{{\\text{{adam}}}} = {eps_val:.1e}$
- **Grow Rule:** $\\sigma \\leftarrow \\min(\\sigma_{{\\max}},\\sigma \\cdot {grow_factor:.2f})$ after `{accept_streak_to_grow}` consecutive accepts
- **Shrink Rule:** $\\sigma \\leftarrow \\max(\\sigma_{{\\min}},\\sigma \\cdot {shrink_factor:.2f})$ after `{reject_streak_to_shrink}` consecutive rejects
- **Step Bounds:** $\\sigma_{{\\min}} = {min_step_size:.1e}$, $\\sigma_{{\\max}} = {upper_sigma_str}$
"""
            )
        else:
            p_col2.markdown(
                f"""
- **Step Adaptation:** Disabled (`Simple Hill Climbing` keeps $\\sigma_t = \\sigma_0 = {step_size:.4g}$ fixed)
- **Momentum Drift:** Disabled (`drift = 0` at every iteration)
- **Stop Budget:** `max_evals = {max_evals}` · `patience = {max_no_improve}`
"""
            )

    # Dynamic Plot at Eval #walk_step
    if dim == 1:
        fig_walk = plot_step_walkthrough_1d(
            spec=spec,
            res=chosen_walk,
            bounds_x=(lo_val, hi_val),
            eval_step=walk_step,
            fog_of_war=walk_fog,
            neighborhood_name=neighborhood_name,
            zoom_to_window=walk_zoom,
        )
    else:
        fig_walk = plot_step_walkthrough_2d(
            spec=spec,
            res=chosen_walk,
            bounds_x=(lo_val, hi_val),
            bounds_y=(lo_val, hi_val),
            eval_step=walk_step,
            fog_of_war=walk_fog,
            neighborhood_name=neighborhood_name,
            zoom_to_window=walk_zoom,
        )
    st.plotly_chart(fig_walk, width="stretch", theme=None if walk_fog else "streamlit")

    # Current evaluation record
    rec = chosen_walk.history[walk_step - 1]
    sigma_in = float(rec.step_size_before if rec.step_size_before is not None else rec.step_size)
    sigma_out = float(rec.step_size)

    # KPI Row
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric(
        "Evaluation & Outcome",
        f"Eval #{rec.eval_index} / {max_eval_w}",
        delta=rec.step_event,
        delta_color="normal" if rec.accepted else "off",
    )
    k2.metric("Candidate x_new", _fmt_vec(rec.x, decimals=4))
    if rec.eval_index > 1 and rec.prev_f is not None:
        df_val = rec.f - rec.prev_f
        k3.metric(
            "f(x_new) vs Anchor",
            _fmt_num(rec.f, decimals=5),
            delta=f"{df_val:+.5f} ({'better' if rec.accepted else 'not better'})",
            delta_color="inverse",
        )
    else:
        k3.metric("Initial f(x₀)", _fmt_num(rec.f, decimals=5), delta="Baseline anchor", delta_color="off")
    k4.metric(
        "Best Found f(x_best)",
        _fmt_num(rec.best_f, decimals=5),
        delta=f"x = {_fmt_vec(rec.best_x, decimals=4)}",
        delta_color="off",
    )
    sigma_delta_text = (
        f"from {_fmt_num(sigma_in, decimals=4)}"
        if abs(sigma_out - sigma_in) > 1e-15
        else "unchanged"
    )
    k5.metric("Step Size σ_after", _fmt_num(sigma_out, decimals=4), delta=sigma_delta_text, delta_color="off")

    st.divider()
    head_c1, head_c2 = st.columns([3.2, 1.4])
    head_c1.subheader(
        f"Step-by-Step Calculations — Inspecting Eval #{rec.eval_index} of {max_eval_w} (Iter {rec.iter_index})"
    )
    head_c1.caption(
        "Anchor `x_curr` always equals the best point found (`best_x`); rejected candidates never overwrite it. "
        "Use the slider above, open the Pop-up for any evaluation, or click any row in the Cumulative History table."
    )
    if head_c2.button(
        f"Open Pop-up (Eval #{rec.eval_index} / Switch Any)",
        key="walk_open_cur_popup",
        width="stretch",
    ):
        st.session_state["walk_popup_eval"] = int(rec.eval_index)
        st.session_state["walk_popup_open"] = True

    _render_eval_step_details(
        chosen_walk,
        rec,
        dim=dim,
        bounds=bounds,
        lo_val=lo_val,
        hi_val=hi_val,
        neighborhood_name=neighborhood_name,
        boundary_mode=boundary_mode,
        seed_val=seed,
        momentum_gain=momentum_gain,
        beta1=beta1,
        beta2=beta2,
        reject_momentum_decay=reject_momentum_decay,
        grow_factor=float(grow_factor),
        shrink_factor=float(shrink_factor),
        accept_streak_to_grow=accept_streak_to_grow,
        reject_streak_to_shrink=reject_streak_to_shrink,
        min_step_size=min_step_size,
    )

    st.divider()

    # Tables: Before -> After Variable State & Cumulative Execution Log (with Pop-up on row click)
    tbl_col1, tbl_col2 = st.columns([1.0, 1.55])
    with tbl_col1:
        st.subheader(f"State Transition (Eval #{rec.eval_index})")
        state_rows = _build_state_transition_rows(
            rec,
            is_adaptive=is_adaptive_walk,
            reject_momentum_decay=reject_momentum_decay,
            accept_streak_to_grow=accept_streak_to_grow,
            reject_streak_to_shrink=reject_streak_to_shrink,
        )
        st.dataframe(pd.DataFrame(state_rows), hide_index=True, width="stretch")

    with tbl_col2:
        cum_head1, cum_head2, cum_head3 = st.columns([2.2, 1.3, 1.3])
        show_all_rows = cum_head2.checkbox(
            f"Show all {max_eval_w} evals",
            value=False,
            key="walk_show_all_rows",
            help="When unchecked, table shows Evals 1..replay step so it matches the plot above.",
        )
        rows_limit = max_eval_w if show_all_rows else walk_step
        cum_head1.subheader(f"Cumulative History (Evals 1 – {rows_limit} of {max_eval_w})")

        if "walk_popup_picker" not in st.session_state:
            st.session_state["walk_popup_picker"] = 1
        st.session_state["walk_popup_picker"] = max(
            1, min(max_eval_w, int(st.session_state["walk_popup_picker"]))
        )

        p_col_sel, p_col_btn = cum_head3.columns([1.0, 1.3])
        picked_pop_eval = int(
            p_col_sel.number_input(
                "Eval #",
                min_value=1,
                max_value=max_eval_w,
                step=1,
                key="walk_popup_picker",
                label_visibility="collapsed",
            )
        )
        if p_col_btn.button("Pop-up", key="walk_open_picked_popup", width="stretch"):
            st.session_state["walk_popup_eval"] = picked_pop_eval
            st.session_state["walk_popup_open"] = True

        st.caption(
            "Click **any row** below (or pick an `Eval #` and click **Pop-up**) to view full step-by-step calculations for that evaluation. "
            "Notice on each row: `anchor_x + drift + offset ε = x_new` (before bounds)."
        )

        cum_rows = []
        for r in chosen_walk.history[:rows_limit]:
            s_in = float(r.step_size_before if r.step_size_before is not None else r.step_size)
            row_dict = {
                "Eval": r.eval_index,
                "Event": r.step_event,
                "anchor_x": _fmt_vec(r.prev_x, decimals=4),
                "σ_in": _fmt_num(s_in, decimals=4),
                "drift": _fmt_vec(r.drift, signed=True, decimals=4) if is_adaptive_walk else "0",
                "offset ε": _fmt_vec(r.offset, signed=True, decimals=4),
                "x_new": _fmt_vec(r.x, decimals=4),
                "f(x_new)": _fmt_num(r.f, decimals=5),
                "acc_strk": r.accept_streak_after,
                "rej_strk": r.reject_streak_after,
                "σ_after": _fmt_num(r.step_size, decimals=4),
                "best_x": _fmt_vec(r.best_x, decimals=4),
                "best_f": _fmt_num(r.best_f, decimals=5),
            }
            if is_adaptive_walk:
                row_dict["m_after"] = _fmt_vec(r.m_after, decimals=4)
                row_dict["v_after"] = _fmt_vec(r.v_after, decimals=4)
            cum_rows.append(row_dict)

        cum_event = st.dataframe(
            pd.DataFrame(cum_rows),
            hide_index=True,
            width="stretch",
            key="walk_cum_df",
            on_select="rerun",
            selection_mode="single-row",
        )
        sel_rows = (
            list(cum_event.selection.rows)
            if cum_event is not None and hasattr(cum_event, "selection")
            else []
        )
        prev_sel = st.session_state.get("_walk_cum_prev_sel", [])
        if sel_rows != prev_sel:
            st.session_state["_walk_cum_prev_sel"] = sel_rows
            if len(sel_rows) == 1:
                r_idx = int(sel_rows[0])
                if 0 <= r_idx < len(cum_rows):
                    st.session_state["walk_popup_eval"] = int(cum_rows[r_idx]["Eval"])
                    st.session_state["walk_popup_open"] = True

    with st.expander("Browse Full Calculations Across Multiple Evaluations (Sequential Range)", expanded=False):
        st.caption(
            "Inspect full Phase 1 → Phase 2 → Phase 3 equations for several consecutive evaluations stacked together "
            "(up to 8 evaluations at a time)."
        )
        if max_eval_w > 1:
            if "walk_multi_start" not in st.session_state:
                st.session_state["walk_multi_start"] = 1
            st.session_state["walk_multi_start"] = max(
                1, min(max_eval_w, int(st.session_state["walk_multi_start"]))
            )
            b_c1, b_c2 = st.columns(2)
            range_start = int(
                b_c1.number_input(
                    "From Eval #",
                    min_value=1,
                    max_value=max_eval_w,
                    step=1,
                    key="walk_multi_start",
                )
            )
            max_end = min(max_eval_w, range_start + 7)
            if "walk_multi_end" not in st.session_state:
                st.session_state["walk_multi_end"] = min(max_end, range_start + 3)
            st.session_state["walk_multi_end"] = max(
                range_start, min(max_end, int(st.session_state["walk_multi_end"]))
            )
            range_end = int(
                b_c2.number_input(
                    "To Eval # (max 8 per batch)",
                    min_value=range_start,
                    max_value=max_end,
                    step=1,
                    key="walk_multi_end",
                )
            )
        else:
            range_start, range_end = 1, 1

        for r_item in chosen_walk.history[range_start - 1 : range_end]:
            with st.container(border=True):
                st.markdown(f"### Eval #{r_item.eval_index} — `{r_item.step_event}`")
                _render_eval_step_details(
                    chosen_walk,
                    r_item,
                    dim=dim,
                    bounds=bounds,
                    lo_val=lo_val,
                    hi_val=hi_val,
                    neighborhood_name=neighborhood_name,
                    boundary_mode=boundary_mode,
                    seed_val=seed,
                    momentum_gain=momentum_gain,
                    beta1=beta1,
                    beta2=beta2,
                    reject_momentum_decay=reject_momentum_decay,
                    grow_factor=float(grow_factor),
                    shrink_factor=float(shrink_factor),
                    accept_streak_to_grow=accept_streak_to_grow,
                    reject_streak_to_shrink=reject_streak_to_shrink,
                    min_step_size=min_step_size,
                )

    @st.dialog(
        f"Full Step-by-Step Calculations — {walk_algo_name}",
        width="large",
        on_dismiss=_close_walk_popup,
    )
    def _show_walk_popup_dialog() -> None:
        cur_p = int(st.session_state.get("walk_popup_eval", 1))
        st.session_state["walk_popup_eval"] = max(1, min(max_eval_w, cur_p))

        m1, m2, m3, m4 = st.columns([0.9, 2.6, 0.9, 1.0])
        m1.button(
            "◀ Prev",
            key="walk_pop_prev",
            width="stretch",
            disabled=(st.session_state["walk_popup_eval"] <= 1),
            on_click=_walk_pop_delta,
            args=(-1, max_eval_w),
        )
        pop_eval = int(
            m2.selectbox(
                "Evaluation inside Pop-up",
                options=list(range(1, max_eval_w + 1)),
                format_func=lambda idx: f"Eval #{idx} — {chosen_walk.history[idx - 1].step_event}",
                key="walk_popup_eval",
                label_visibility="collapsed",
            )
        )
        m3.button(
            "Next ▶",
            key="walk_pop_next",
            width="stretch",
            disabled=(pop_eval >= max_eval_w),
            on_click=_walk_pop_delta,
            args=(1, max_eval_w),
        )
        m4.button(
            "Close",
            key="walk_pop_close",
            width="stretch",
            on_click=_close_walk_popup,
        )

        p_rec = chosen_walk.history[pop_eval - 1]
        p_sig_in = float(p_rec.step_size_before if p_rec.step_size_before is not None else p_rec.step_size)
        p_sig_out = float(p_rec.step_size)

        pk1, pk2, pk3, pk4 = st.columns(4)
        pk1.metric("Eval & Outcome", f"#{p_rec.eval_index} / {max_eval_w}", delta=p_rec.step_event, delta_color="off")
        pk2.metric("Candidate x_new", _fmt_vec(p_rec.x, decimals=4))
        pk3.metric("Best f(x_best)", _fmt_num(p_rec.best_f, decimals=5), delta=f"x={_fmt_vec(p_rec.best_x, decimals=4)}", delta_color="off")
        pk4.metric("σ_in → σ_out", f"{_fmt_num(p_sig_in, decimals=4)} → {_fmt_num(p_sig_out, decimals=4)}")

        st.divider()
        _render_eval_step_details(
            chosen_walk,
            p_rec,
            dim=dim,
            bounds=bounds,
            lo_val=lo_val,
            hi_val=hi_val,
            neighborhood_name=neighborhood_name,
            boundary_mode=boundary_mode,
            seed_val=seed,
            momentum_gain=momentum_gain,
            beta1=beta1,
            beta2=beta2,
            reject_momentum_decay=reject_momentum_decay,
            grow_factor=float(grow_factor),
            shrink_factor=float(shrink_factor),
            accept_streak_to_grow=accept_streak_to_grow,
            reject_streak_to_shrink=reject_streak_to_shrink,
            min_step_size=min_step_size,
        )
        st.divider()
        st.subheader(f"State Transition Summary (Eval #{p_rec.eval_index})")
        pop_state_rows = _build_state_transition_rows(
            p_rec,
            is_adaptive=is_adaptive_walk,
            reject_momentum_decay=reject_momentum_decay,
            accept_streak_to_grow=accept_streak_to_grow,
            reject_streak_to_shrink=reject_streak_to_shrink,
        )
        st.dataframe(pd.DataFrame(pop_state_rows), hide_index=True, width="stretch")

    if st.session_state.get("walk_popup_open", False):
        _show_walk_popup_dialog()

    with st.expander("Parameter Influence Guide (Try Adjusting Sidebar Sliders Live)", expanded=False):
        st.markdown(
            """
- **Initial step size $\\sigma_0$:** Sets both the fixed radius of `Simple HC` and the starting scale of `drift` and random probes in `Adaptive HC`. Too small traps early in ripples; too large overshoots narrow valleys.
- **Momentum gain $\\eta_{\\text{mom}}$ & $\\beta_1, \\beta_2$:** Once $t_{\\text{mom}} \\ge 1$, the search center shifts by $\\eta_{\\text{mom}}\\sigma_t d_{\\text{norm}}$ along recent accepted descent directions. Increasing $\\beta_1$ lengthens directional memory; setting $\\eta_{\\text{mom}} = 0$ removes drift while keeping streak scaling.
- **Reject damping $\\gamma_{\\text{reject}}$:** When a probe is rejected, multiplying $m \\leftarrow \\gamma_{\\text{reject}} m$ reduces push along a blocked direction. Smaller $\\gamma_{\\text{reject}}$ cancels wrong-way momentum faster.
- **Accept streak (`c_grow`, `K_grow`) vs. Reject streak (`c_shrink`, `K_shrink`):** Expanding $\\sigma$ after $K_{\\text{grow}}$ accepts accelerates across slopes/plateaus; shrinking $\\sigma$ after $K_{\\text{shrink}}$ rejects zooms into local minima (and hits finer precision near $f^*$).
"""
        )
