"""Interactive Plotly visualizations (including Fog of War mode)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .benchmarks import BenchmarkSpec, evaluate_grid_1d, evaluate_grid_2d
from .hill_climbing import OptimizationResult

ALGO_COLORS = {
    "Simple Hill Climbing": "#2563EB",
    "Adaptive Momentum Hill Climbing": "#EA580C",
}
ALGO_SHORT_NAMES = {
    "Simple Hill Climbing": "Simple HC",
    "Adaptive Momentum Hill Climbing": "Adaptive HC",
}
ALGO_LIGHT_COLORS = {
    "Simple Hill Climbing": "rgba(37, 99, 235, 0.14)",
    "Adaptive Momentum Hill Climbing": "rgba(234, 88, 12, 0.14)",
}


def _short_name(algorithm_name: str) -> str:
    return ALGO_SHORT_NAMES.get(algorithm_name, algorithm_name)


def _slice_records(res: OptimizationResult, max_eval_step: int | None):
    limit = res.eval_count if max_eval_step is None else min(max_eval_step, res.eval_count)
    return [r for r in res.history if r.eval_index <= limit]


def _current_accepted_record(recs):
    accepted = [r for r in recs if r.accepted]
    return accepted[-1] if accepted else recs[0]


def plot_1d_trajectory(
    *,
    spec: BenchmarkSpec,
    results: list[OptimizationResult],
    bounds_x: tuple[float, float],
    fog_of_war: bool = False,
    max_eval_step: int | None = None,
    show_rejected: bool = True,
    show_neighborhood: bool = True,
) -> go.Figure:
    """Plot 1D benchmark landscape (or Fog-of-War observed probes) with search paths."""
    fig = go.Figure()
    low_x, high_x = bounds_x

    if not fog_of_war:
        xs, ys = evaluate_grid_1d(spec.fn, low_x, high_x, num_points=600)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"True {spec.name}",
                line=dict(color="#1E293B", width=2.5),
            )
        )
        fig.add_hline(
            y=spec.global_min_f,
            line_dash="dot",
            line_color="#16A34A",
            annotation_text=f"Global min f* = {spec.global_min_f:g}",
            annotation_position="bottom right",
        )

    for idx, res in enumerate(results):
        recs = _slice_records(res, max_eval_step)
        if not recs:
            continue
        short = _short_name(res.algorithm_name)
        color = ALGO_COLORS.get(res.algorithm_name, "#0F766E")
        fill_color = ALGO_LIGHT_COLORS.get(res.algorithm_name, "rgba(15, 118, 110, 0.15)")

        accepted = [r for r in recs if r.accepted]
        rejected = [r for r in recs if not r.accepted]
        curr = _current_accepted_record(recs)
        last_probe = recs[-1]

        if show_rejected and rejected:
            fig.add_trace(
                go.Scatter(
                    x=[float(r.x[0]) for r in rejected],
                    y=[r.f for r in rejected],
                    mode="markers",
                    name=f"{short}: rejected",
                    marker=dict(
                        symbol="x",
                        size=7,
                        color=color,
                        opacity=0.45,
                    ),
                    customdata=[[r.eval_index, r.step_size] for r in rejected],
                    hovertemplate=(
                        f"<b>{res.algorithm_name} — Rejected</b><br>"
                        "eval #%{customdata[0]}<br>"
                        "x = %{x:.4f}<br>"
                        "f(x) = %{y:.5f}<br>"
                        "σ = %{customdata[1]:.4f}<extra></extra>"
                    ),
                )
            )

        if accepted:
            fig.add_trace(
                go.Scatter(
                    x=[float(r.x[0]) for r in accepted],
                    y=[r.f for r in accepted],
                    mode="lines+markers",
                    name=f"{short}: path",
                    line=dict(color=color, width=2.2),
                    marker=dict(size=7, color=color, line=dict(width=1, color="white")),
                    customdata=[[r.eval_index, r.step_size] for r in accepted],
                    hovertemplate=(
                        f"<b>{res.algorithm_name} — Accepted</b><br>"
                        "eval #%{customdata[0]}<br>"
                        "x = %{x:.4f}<br>"
                        "f(x) = %{y:.5f}<br>"
                        "σ = %{customdata[1]:.4f}<extra></extra>"
                    ),
                )
            )

        start = recs[0]
        fig.add_trace(
            go.Scatter(
                x=[float(start.x[0])],
                y=[start.f],
                mode="markers",
                name=f"{short}: start",
                marker=dict(symbol="diamond", size=11, color=color, line=dict(width=1.5, color="black")),
                showlegend=False,
                hovertemplate=f"Start x0 = {float(start.x[0]):.4f}, f(x0) = {start.f:.5f}<extra></extra>",
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[float(curr.x[0])],
                y=[curr.f],
                mode="markers",
                name=f"{short}: best",
                marker=dict(symbol="star", size=14, color=color, line=dict(width=1.5, color="black")),
                hovertemplate=(
                    f"<b>{res.algorithm_name} current best</b><br>"
                    f"x = {float(curr.x[0]):.5f}<br>"
                    f"f(x) = {curr.f:.6f}<br>"
                    f"σ = {last_probe.step_size:.5f}<extra></extra>"
                ),
            )
        )

        if show_neighborhood:
            center_x = float(curr.x[0])
            if last_probe.drift is not None:
                center_x = float(np.clip(center_x + last_probe.drift[0], low_x, high_x))
            sigma_now = last_probe.step_size
            nb_lo = max(low_x, center_x - sigma_now)
            nb_hi = min(high_x, center_x + sigma_now)
            fig.add_vrect(
                x0=nb_lo,
                x1=nb_hi,
                fillcolor=fill_color,
                line_width=1,
                line_dash="dash",
                line_color=color,
                annotation_text=f"{short} ±σ ({sigma_now:.3g})",
                annotation_position="top left" if idx % 2 == 0 else "bottom left",
                annotation_font=dict(size=11, color=color),
            )

    title_suffix = " [Fog of War]" if fog_of_war else ""
    fig.update_layout(
        title=dict(text=f"{spec.name} Search Trajectory{title_suffix}", x=0.0, xanchor="left"),
        xaxis_title="x",
        yaxis_title="f(x)",
        xaxis=dict(range=[low_x, high_x]),
        template="plotly_dark" if fog_of_war else "plotly_white",
        height=520,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=55, r=25, t=60, b=130),
    )
    return fig


def plot_2d_contour_trajectory(
    *,
    spec: BenchmarkSpec,
    results: list[OptimizationResult],
    bounds_x: tuple[float, float],
    bounds_y: tuple[float, float],
    fog_of_war: bool = False,
    max_eval_step: int | None = None,
    show_rejected: bool = True,
    show_neighborhood: bool = True,
) -> go.Figure:
    """Plot 2D Ackley contour (or Fog-of-War observed probe map) with trajectories."""
    fig = go.Figure()
    low_x, high_x = bounds_x
    low_y, high_y = bounds_y

    if not fog_of_war:
        xs, ys, gz = evaluate_grid_2d(spec.fn, low_x, high_x, low_y, high_y, num_points=150)
        fig.add_trace(
            go.Contour(
                x=xs,
                y=ys,
                z=gz,
                colorscale="Viridis",
                contours=dict(coloring="heatmap", showlabels=False),
                opacity=0.82,
                colorbar=dict(title="f(x₁, x₂)"),
                name="Ackley landscape",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0.0],
                y=[0.0],
                mode="markers",
                name="Global min (0, 0)",
                marker=dict(symbol="x-thin", size=14, color="#22C55E", line=dict(width=3, color="#22C55E")),
            )
        )
    else:
        all_fs = [r.f for res in results for r in _slice_records(res, max_eval_step)]
        cmin = min(all_fs) if all_fs else 0.0
        cmax = max(all_fs) if all_fs else 20.0
        if cmax <= cmin:
            cmax = cmin + 1.0

    for res in results:
        recs = _slice_records(res, max_eval_step)
        if not recs:
            continue
        short = _short_name(res.algorithm_name)
        color = ALGO_COLORS.get(res.algorithm_name, "#0F766E")
        accepted = [r for r in recs if r.accepted]
        rejected = [r for r in recs if not r.accepted]
        curr = _current_accepted_record(recs)
        last_probe = recs[-1]

        if show_rejected and rejected:
            marker_kw: dict = dict(symbol="x", size=7, opacity=0.6)
            if fog_of_war:
                marker_kw.update(
                    color=[r.f for r in rejected],
                    colorscale="Viridis",
                    cmin=cmin,
                    cmax=cmax,
                    showscale=False,
                )
            else:
                marker_kw["color"] = color

            fig.add_trace(
                go.Scatter(
                    x=[float(r.x[0]) for r in rejected],
                    y=[float(r.x[1]) for r in rejected],
                    mode="markers",
                    name=f"{short}: rejected",
                    marker=marker_kw,
                    customdata=[[r.eval_index, r.f, r.step_size] for r in rejected],
                    hovertemplate=(
                        f"<b>{res.algorithm_name} — Rejected</b><br>"
                        "eval #%{customdata[0]}<br>"
                        "x = (%{x:.3f}, %{y:.3f})<br>"
                        "f(x) = %{customdata[1]:.5f}<br>"
                        "σ = %{customdata[2]:.4f}<extra></extra>"
                    ),
                )
            )

        if accepted:
            marker_acc: dict = dict(
                size=7,
                line=dict(width=1.3, color=color if fog_of_war else "white"),
            )
            if fog_of_war:
                marker_acc.update(
                    color=[r.f for r in accepted],
                    colorscale="Viridis",
                    cmin=cmin,
                    cmax=cmax,
                    colorbar=dict(title="Observed f(x)"),
                    showscale=(res is results[0]),
                )
            else:
                marker_acc["color"] = color

            fig.add_trace(
                go.Scatter(
                    x=[float(r.x[0]) for r in accepted],
                    y=[float(r.x[1]) for r in accepted],
                    mode="lines+markers",
                    name=f"{short}: path",
                    line=dict(color=color, width=2.5),
                    marker=marker_acc,
                    customdata=[[r.eval_index, r.f, r.step_size] for r in accepted],
                    hovertemplate=(
                        f"<b>{res.algorithm_name} — Accepted</b><br>"
                        "eval #%{customdata[0]}<br>"
                        "x = (%{x:.3f}, %{y:.3f})<br>"
                        "f(x) = %{customdata[1]:.5f}<br>"
                        "σ = %{customdata[2]:.4f}<extra></extra>"
                    ),
                )
            )

        start = recs[0]
        fig.add_trace(
            go.Scatter(
                x=[float(start.x[0])],
                y=[float(start.x[1])],
                mode="markers",
                name=f"{short}: start",
                marker=dict(symbol="diamond", size=11, color=color, line=dict(width=1.5, color="white")),
                showlegend=False,
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[float(curr.x[0])],
                y=[float(curr.x[1])],
                mode="markers",
                name=f"{short}: best",
                marker=dict(symbol="star", size=14, color=color, line=dict(width=1.5, color="white")),
            )
        )

        if show_neighborhood:
            theta = np.linspace(0.0, 2.0 * np.pi, 90)
            cx = float(curr.x[0])
            cy = float(curr.x[1])
            if last_probe.drift is not None:
                cx = float(np.clip(cx + last_probe.drift[0], low_x, high_x))
                cy = float(np.clip(cy + last_probe.drift[1], low_y, high_y))
                fig.add_trace(
                    go.Scatter(
                        x=[float(curr.x[0]), cx],
                        y=[float(curr.x[1]), cy],
                        mode="lines",
                        line=dict(color=color, width=2, dash="dot"),
                        name=f"{short}: drift",
                        showlegend=False,
                    )
                )
            sigma_now = last_probe.step_size
            fig.add_trace(
                go.Scatter(
                    x=cx + sigma_now * np.cos(theta),
                    y=cy + sigma_now * np.sin(theta),
                    mode="lines",
                    line=dict(color=color, width=1.6, dash="dash"),
                    name=f"{short}: σ",
                )
            )

    title_suffix = " [Fog of War]" if fog_of_war else ""
    fig.update_layout(
        title=dict(text=f"{spec.name} Search Map{title_suffix}", x=0.0, xanchor="left"),
        xaxis_title="x₁",
        yaxis_title="x₂",
        xaxis=dict(range=[low_x, high_x], scaleanchor="y", scaleratio=1, constrain="domain"),
        yaxis=dict(range=[low_y, high_y], constrain="domain"),
        template="plotly_dark" if fog_of_war else "plotly_white",
        height=630,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.20,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=55, r=25, t=60, b=155),
    )
    return fig


def plot_3d_surface_trajectory(
    *,
    spec: BenchmarkSpec,
    results: list[OptimizationResult],
    bounds_x: tuple[float, float],
    bounds_y: tuple[float, float],
    fog_of_war: bool = False,
    max_eval_step: int | None = None,
    show_rejected: bool = True,
) -> go.Figure:
    """Plot 3D Ackley surface (or Fog-of-War 3D probe cloud) with trajectories."""
    fig = go.Figure()
    low_x, high_x = bounds_x
    low_y, high_y = bounds_y

    if not fog_of_war:
        xs, ys, gz = evaluate_grid_2d(spec.fn, low_x, high_x, low_y, high_y, num_points=95)
        fig.add_trace(
            go.Surface(
                x=xs,
                y=ys,
                z=gz,
                colorscale="Viridis",
                opacity=0.68,
                showscale=False,
                name="Ackley surface",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0.0],
                y=[0.0],
                z=[0.0],
                mode="markers",
                name="Global min (0, 0, 0)",
                marker=dict(symbol="cross", size=6, color="#22C55E"),
            )
        )

    for res in results:
        recs = _slice_records(res, max_eval_step)
        if not recs:
            continue
        short = _short_name(res.algorithm_name)
        color = ALGO_COLORS.get(res.algorithm_name, "#0F766E")
        accepted = [r for r in recs if r.accepted]
        rejected = [r for r in recs if not r.accepted]
        curr = _current_accepted_record(recs)

        if show_rejected and rejected:
            fig.add_trace(
                go.Scatter3d(
                    x=[float(r.x[0]) for r in rejected],
                    y=[float(r.x[1]) for r in rejected],
                    z=[r.f for r in rejected],
                    mode="markers",
                    name=f"{short}: rejected",
                    marker=dict(size=3, symbol="x", color=color, opacity=0.45),
                )
            )

        if accepted:
            fig.add_trace(
                go.Scatter3d(
                    x=[float(r.x[0]) for r in accepted],
                    y=[float(r.x[1]) for r in accepted],
                    z=[r.f for r in accepted],
                    mode="lines+markers",
                    name=f"{short}: path",
                    line=dict(color=color, width=5),
                    marker=dict(size=4, color=color),
                )
            )

        fig.add_trace(
            go.Scatter3d(
                x=[float(curr.x[0])],
                y=[float(curr.x[1])],
                z=[curr.f],
                mode="markers",
                name=f"{short}: best",
                marker=dict(size=7, symbol="diamond", color=color, line=dict(width=2, color="white")),
            )
        )

    title_suffix = " [Fog of War]" if fog_of_war else ""
    fig.update_layout(
        title=dict(text=f"{spec.name} 3D View{title_suffix}", x=0.0, xanchor="left"),
        scene=dict(
            xaxis=dict(title="x₁", range=[low_x, high_x]),
            yaxis=dict(title="x₂", range=[low_y, high_y]),
            zaxis=dict(title="f(x)"),
        ),
        template="plotly_dark" if fog_of_war else "plotly_white",
        height=610,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.06,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(l=15, r=15, t=60, b=120),
    )
    return fig


def plot_convergence_and_step_size(
    *,
    results: list[OptimizationResult],
    global_min_f: float,
    max_eval_step: int | None = None,
    log_gap: bool = False,
) -> go.Figure:
    """Plot best-so-far objective value and adaptive step size vs objective evaluations."""
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        subplot_titles=(
            "Optimality Gap |f(x_best) - f*| vs. Evaluations" if log_gap else "Best Objective Value f(x_best) vs. Evaluations",
            "Neighborhood Step Size σ vs. Evaluations",
        ),
    )

    for res in results:
        recs = _slice_records(res, max_eval_step)
        if not recs:
            continue
        color = ALGO_COLORS.get(res.algorithm_name, "#0F766E")
        evals = [r.eval_index for r in recs]
        y_vals = (
            [max(abs(r.best_f - global_min_f), 1e-12) for r in recs]
            if log_gap
            else [r.best_f for r in recs]
        )
        sigmas = [r.step_size for r in recs]

        fig.add_trace(
            go.Scatter(
                x=evals,
                y=y_vals,
                mode="lines",
                name=res.algorithm_name,
                line=dict(color=color, width=2.5),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=evals,
                y=sigmas,
                mode="lines",
                name=f"{res.algorithm_name} (σ)",
                line=dict(color=color, width=2.2, dash="solid"),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Objective function evaluations", row=1, col=1)
    fig.update_xaxes(title_text="Objective function evaluations", row=1, col=2)
    fig.update_yaxes(
        title_text="|f(x_best) - f*|" if log_gap else "Best f(x)",
        type="log" if log_gap else "linear",
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Step size σ", type="log", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        height=430,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=55, r=25, t=55, b=105),
    )
    return fig


def plot_batch_distribution(runs_df: pd.DataFrame, global_min_f: float) -> go.Figure:
    """Plot multi-start final objective quality and evaluation count distributions."""
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        subplot_titles=("Final Best f(x) Across Paired Random Starts", "Evaluations Used vs. Final Gap"),
    )

    for algo in runs_df["algorithm"].unique():
        sub = runs_df[runs_df["algorithm"] == algo]
        color = ALGO_COLORS.get(algo, "#0F766E")

        fig.add_trace(
            go.Box(
                y=sub["best_f"],
                name=algo,
                marker_color=color,
                boxpoints="all",
                jitter=0.3,
                pointpos=-1.5,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=sub["eval_count"],
                y=np.maximum(sub["optimality_gap"], 1e-12),
                mode="markers",
                name=algo,
                marker=dict(color=color, size=8, opacity=0.8),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.add_hline(y=global_min_f, line_dash="dot", line_color="#16A34A", row=1, col=1)
    fig.update_yaxes(title_text="Final best f(x)", row=1, col=1)
    fig.update_xaxes(title_text="Objective evaluations", row=1, col=2)
    fig.update_yaxes(title_text="|best_f - f*|", type="log", row=1, col=2)
    fig.update_layout(
        template="plotly_white",
        height=450,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=55, r=25, t=55, b=105),
    )
    return fig
