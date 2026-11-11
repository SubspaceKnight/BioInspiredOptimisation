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
# Main Page Sections (PDF 1.3 required structure)
# ---------------------------------------------------------------------------
tab_intro, tab_methods, tab_results, tab_discussion = st.tabs(
    ["1. Introduction", "2. Methods", "3. Results", "4. Discussion"]
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
    replay_step = ctrl5.slider(
        "Evaluation replay step",
        min_value=1,
        max_value=max_recorded_eval,
        value=max_recorded_eval,
        step=1,
        help="Scrub evaluation-by-evaluation to observe exploration and local-minimum trapping.",
    )

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
