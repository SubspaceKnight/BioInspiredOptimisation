"""Tests for benchmarks, Hill Climbing variants, experiments, and Fog-of-War plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bio_inspired_optimisation.benchmarks import (
    BENCHMARKS,
    ackley,
    evaluate_grid_1d,
    evaluate_grid_2d,
    quadratic,
    sinusoidal,
)
from bio_inspired_optimisation.experiments import run_batch_benchmark, run_single_comparison
from bio_inspired_optimisation.hill_climbing import (
    adaptive_momentum_hill_climbing,
    simple_hill_climbing,
)
from bio_inspired_optimisation.neighborhoods import (
    apply_bounds,
    make_gaussian_neighborhood,
    make_uniform_ball_neighborhood,
    make_uniform_box_neighborhood,
)
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


def test_benchmark_values() -> None:
    assert quadratic(0.0) == pytest.approx(0.0)
    assert quadratic(np.array([3.0, -4.0])) == pytest.approx(25.0)

    assert sinusoidal(0.0) == pytest.approx(0.0)
    assert sinusoidal(-0.5 * np.pi) == pytest.approx(-1.0)

    assert ackley(np.zeros(2)) == pytest.approx(0.0, abs=1e-12)
    assert ackley(np.zeros(5)) == pytest.approx(0.0, abs=1e-12)
    assert ackley(np.array([1.0, 1.0])) > 0.0

    xs, ys = evaluate_grid_1d(quadratic, -2.0, 2.0, num_points=21)
    assert xs.shape == (21,)
    assert ys[0] == pytest.approx(4.0)

    gx, gy, gz = evaluate_grid_2d(ackley, -2.0, 2.0, -2.0, 2.0, num_points=15)
    assert gx.shape == (15,)
    assert gy.shape == (15,)
    assert gz.shape == (15, 15)


def test_neighborhoods_and_bounds() -> None:
    rng = np.random.default_rng(7)
    bounds = (np.array([-2.0, -2.0]), np.array([2.0, 2.0]))

    clipped = apply_bounds(np.array([-3.0, 2.5]), bounds, mode="clip")
    assert np.allclose(clipped, [-2.0, 2.0])

    reflected = apply_bounds(np.array([-2.5, 2.4]), bounds, mode="reflect")
    assert np.allclose(reflected, [-1.5, 1.6])

    for factory in (
        make_uniform_box_neighborhood,
        make_uniform_ball_neighborhood,
        make_gaussian_neighborhood,
    ):
        nb = factory("clip")
        for _ in range(30):
            cand = nb(np.array([0.5, -0.5]), 0.8, rng, bounds=bounds)
            assert cand.shape == (2,)
            assert np.all(cand >= bounds[0]) and np.all(cand <= bounds[1])

    ball_nb = make_uniform_ball_neighborhood("clip")
    for _ in range(30):
        cand = ball_nb(np.zeros(2), 0.5, rng, bounds=None)
        assert np.linalg.norm(cand) <= 0.5 + 1e-12


def test_termination_criterion() -> None:
    term = TerminationCriterion(
        max_evals=100,
        max_iters=40,
        max_no_improve=15,
        min_step_size=1e-4,
        target_fitness=1e-3,
    )
    assert term.check(eval_count=10, iter_count=9, no_improve_count=0, best_f=5e-4, current_step_size=0.1) == "target_fitness"
    assert term.check(eval_count=100, iter_count=20, no_improve_count=0, best_f=1.0, current_step_size=0.1) == "max_evals"
    assert term.check(eval_count=20, iter_count=40, no_improve_count=0, best_f=1.0, current_step_size=0.1) == "max_iters"
    assert term.check(eval_count=20, iter_count=19, no_improve_count=15, best_f=1.0, current_step_size=0.1) == "max_no_improve"
    assert term.check(eval_count=20, iter_count=19, no_improve_count=2, best_f=1.0, current_step_size=1e-4) == "min_step_size"
    assert term.check(eval_count=20, iter_count=19, no_improve_count=2, best_f=1.0, current_step_size=0.1) is None


def test_exact_eval_counting_and_monotonic_descent() -> None:
    calls = 0

    def counted_ackley(x: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return ackley(x)

    nb = make_gaussian_neighborhood("clip")
    term = TerminationCriterion(max_evals=75)

    res1 = simple_hill_climbing([4.0, -3.5], counted_ackley, nb, term, step_size=0.8, seed=11)
    assert res1.eval_count == calls == len(res1.history) == 75

    calls = 0
    res2 = adaptive_momentum_hill_climbing([4.0, -3.5], counted_ackley, nb, term, initial_step_size=0.8, seed=11)
    assert res2.eval_count == calls == len(res2.history) == 75

    for res in (res1, res2):
        best_curve = res.best_f_curve()
        assert np.all(np.diff(best_curve) <= 1e-15)
        assert res.best_f <= res.initial_f


def test_reproducibility_and_custom_external_neighborhood() -> None:
    def external_nb(x: np.ndarray, step_size: float, rng: np.random.Generator, bounds=None) -> np.ndarray:
        return x + rng.uniform(-step_size, step_size, size=x.shape)

    term = TerminationCriterion(max_evals=60)
    r_a = adaptive_momentum_hill_climbing([5.0], quadratic, external_nb, term, initial_step_size=0.5, seed=99)
    r_b = adaptive_momentum_hill_climbing([5.0], quadratic, external_nb, term, initial_step_size=0.5, seed=99)
    assert r_a.best_f == pytest.approx(r_b.best_f)
    assert np.allclose(r_a.best_x, r_b.best_x)


def test_maximization_mode() -> None:
    nb = make_uniform_box_neighborhood("clip")
    bounds = (np.array([-2.0]), np.array([2.0]))
    term = TerminationCriterion(max_evals=100)
    res = simple_hill_climbing(
        [0.0],
        lambda x: -float(np.sum(x**2)),
        nb,
        term,
        step_size=0.4,
        bounds=bounds,
        minimize=False,
        seed=5,
    )
    assert res.best_f >= res.initial_f

    res_adapt = adaptive_momentum_hill_climbing(
        [-1.0],
        sinusoidal,
        nb,
        term,
        initial_step_size=0.5,
        bounds=(np.array([-np.pi]), np.array([np.pi])),
        minimize=False,
        seed=5,
    )
    assert res_adapt.best_f > 0.9


def test_adaptive_momentum_precision_on_quadratic() -> None:
    nb = make_uniform_box_neighborhood("clip")
    bounds = BENCHMARKS["quadratic"].make_bounds(dim=1)
    term = TerminationCriterion(max_evals=150)

    simple_res = simple_hill_climbing([8.0], quadratic, nb, term, step_size=0.6, bounds=bounds, seed=42)
    adapt_res = adaptive_momentum_hill_climbing(
        [8.0],
        quadratic,
        nb,
        term,
        initial_step_size=0.6,
        bounds=bounds,
        seed=42,
    )
    assert adapt_res.best_f < 1e-4
    assert adapt_res.best_f < simple_res.best_f


def test_experiments_and_fog_of_war_plots() -> None:
    spec_1d = BENCHMARKS["sinusoidal"]
    bounds_1d = spec_1d.make_bounds(dim=1)
    term = TerminationCriterion(max_evals=60)
    s1, a1 = run_single_comparison(
        spec=spec_1d,
        initial_solution=np.array([-4.0]),
        bounds=bounds_1d,
        neighborhood_name="Uniform Box [-δ, δ]ⁿ",
        boundary_mode="clip",
        step_size=0.5,
        termination=term,
        seed=42,
    )

    fig_1d_clear = plot_1d_trajectory(
        spec=spec_1d,
        results=[s1, a1],
        bounds_x=(float(bounds_1d[0][0]), float(bounds_1d[1][0])),
        fog_of_war=False,
    )
    fig_1d_fog = plot_1d_trajectory(
        spec=spec_1d,
        results=[s1, a1],
        bounds_x=(float(bounds_1d[0][0]), float(bounds_1d[1][0])),
        fog_of_war=True,
        max_eval_step=25,
    )
    assert any(tr.name and tr.name.startswith("True ") for tr in fig_1d_clear.data)
    assert not any(tr.name and tr.name.startswith("True ") for tr in fig_1d_fog.data)

    spec_2d = BENCHMARKS["ackley"]
    bounds_2d = spec_2d.make_bounds(dim=2)
    s2, a2 = run_single_comparison(
        spec=spec_2d,
        initial_solution=np.array([-8.0, 7.0]),
        bounds=bounds_2d,
        neighborhood_name="Gaussian N(0, δ²I)",
        boundary_mode="clip",
        step_size=1.2,
        termination=term,
        seed=7,
    )
    bx = (float(bounds_2d[0][0]), float(bounds_2d[1][0]))

    fig_2d_clear = plot_2d_contour_trajectory(spec=spec_2d, results=[s2, a2], bounds_x=bx, bounds_y=bx, fog_of_war=False)
    fig_2d_fog = plot_2d_contour_trajectory(spec=spec_2d, results=[s2, a2], bounds_x=bx, bounds_y=bx, fog_of_war=True, max_eval_step=20)
    assert any(tr.type == "contour" for tr in fig_2d_clear.data)
    assert not any(tr.type == "contour" for tr in fig_2d_fog.data)

    fig_3d_clear = plot_3d_surface_trajectory(spec=spec_2d, results=[s2, a2], bounds_x=bx, bounds_y=bx, fog_of_war=False)
    fig_3d_fog = plot_3d_surface_trajectory(spec=spec_2d, results=[s2, a2], bounds_x=bx, bounds_y=bx, fog_of_war=True, max_eval_step=20)
    assert any(tr.type == "surface" for tr in fig_3d_clear.data)
    assert not any(tr.type == "surface" for tr in fig_3d_fog.data)

    fig_conv = plot_convergence_and_step_size(results=[s2, a2], global_min_f=0.0, max_eval_step=30, log_gap=True)
    assert len(fig_conv.data) == 4

    runs_df, agg_df = run_batch_benchmark(
        spec=spec_1d,
        dim=1,
        bounds=bounds_1d,
        n_runs=6,
        neighborhood_name="Uniform Box [-δ, δ]ⁿ",
        boundary_mode="clip",
        step_size=0.5,
        termination=term,
        base_seed=10,
    )
    assert len(runs_df) == 12
    assert len(agg_df) == 2
    fig_batch = plot_batch_distribution(runs_df, spec_1d.global_min_f)
    assert len(fig_batch.data) == 4


def test_evaluation_record_trace_and_walkthrough_plots() -> None:
    spec_1d = BENCHMARKS["quadratic"]
    bounds_1d = spec_1d.make_bounds(dim=1)
    term = TerminationCriterion(max_evals=40)
    simple_r, adapt_r = run_single_comparison(
        spec=spec_1d,
        initial_solution=np.array([7.5]),
        bounds=bounds_1d,
        neighborhood_name="Uniform Box [-δ, δ]ⁿ",
        boundary_mode="clip",
        step_size=0.6,
        termination=term,
        seed=42,
    )

    assert simple_r.params["step_size"] == pytest.approx(0.6)
    assert adapt_r.params["initial_step_size"] == pytest.approx(0.6)
    assert adapt_r.history[0].step_event == "Init"
    assert adapt_r.history[0].t_mom_after == 0

    for res in (simple_r, adapt_r):
        for rec in res.history[1:]:
            assert rec.prev_x is not None
            assert rec.prev_f is not None
            assert rec.offset is not None
            assert rec.step_size_before is not None
            drift_vec = np.zeros_like(rec.x) if rec.drift is None else rec.drift
            assert np.allclose(rec.prev_x + drift_vec + rec.offset, rec.x, atol=1e-12)

    events = [r.step_event for r in adapt_r.history]
    assert "Accept (Grow σ)" in events
    assert "Reject (Shrink σ)" in events

    bx_1d = (float(bounds_1d[0][0]), float(bounds_1d[1][0]))
    grow_idx = next(r.eval_index for r in adapt_r.history if r.step_event == "Accept (Grow σ)")
    shrink_idx = next(r.eval_index for r in adapt_r.history if r.step_event == "Reject (Shrink σ)")

    for step in (1, grow_idx, shrink_idx):
        for fog in (False, True):
            fig = plot_step_walkthrough_1d(
                spec=spec_1d,
                res=adapt_r,
                bounds_x=bx_1d,
                eval_step=step,
                fog_of_war=fog,
                neighborhood_name="Uniform Box [-δ, δ]ⁿ",
                zoom_to_window=True,
            )
            assert any(tr.name and tr.name.startswith("True ") for tr in fig.data) is (not fog)

    spec_2d = BENCHMARKS["ackley"]
    bounds_2d = spec_2d.make_bounds(dim=2)
    _, adapt_2d = run_single_comparison(
        spec=spec_2d,
        initial_solution=np.array([-3.0, 2.5]),
        bounds=bounds_2d,
        neighborhood_name="Uniform Box [-δ, δ]ⁿ",
        boundary_mode="clip",
        step_size=1.0,
        termination=term,
        seed=42,
    )
    bx_2d = (float(bounds_2d[0][0]), float(bounds_2d[1][0]))
    fig_2d_box = plot_step_walkthrough_2d(
        spec=spec_2d,
        res=adapt_2d,
        bounds_x=bx_2d,
        bounds_y=bx_2d,
        eval_step=3,
        fog_of_war=False,
        neighborhood_name="Uniform Box [-δ, δ]ⁿ",
        zoom_to_window=True,
    )
    assert any(tr.name and tr.name.startswith("Search box") for tr in fig_2d_box.data)
    fig_2d_fog = plot_step_walkthrough_2d(
        spec=spec_2d,
        res=adapt_2d,
        bounds_x=bx_2d,
        bounds_y=bx_2d,
        eval_step=1,
        fog_of_war=True,
        neighborhood_name="Gaussian N(0, δ²I)",
        zoom_to_window=False,
    )
    assert not any(tr.type == "contour" for tr in fig_2d_fog.data)


def test_streamlit_multipage_app_renders() -> None:
    from streamlit.testing.v1 import AppTest

    root = Path(__file__).resolve().parents[1]
    at_home = AppTest.from_file(root / "Home.py", default_timeout=20).run()
    assert not at_home.exception

    at_ex1 = at_home.switch_page("pages/1_Exercise_01_Hill_Climbing.py").run()
    assert not at_ex1.exception
    assert len(at_ex1.tabs) == 5
    assert at_ex1.tabs[4].label == "5. Step-by-Step Dynamics"

    # Tab 3 Fog of War remains toggle[0] and shows 1 warning
    at_ex1.toggle[0].set_value(True).run()
    assert not at_ex1.exception
    assert len(at_ex1.warning) == 1

    # Switch to 1D quadratic and exercise Tab 5 dynamic walkthrough controls
    at_ex1.sidebar.selectbox[0].select("quadratic").run()
    assert not at_ex1.exception

    step_before = int(at_ex1.slider(key="walkthrough_step").value)
    at_ex1.button(key="walk_next").click().run()
    assert not at_ex1.exception
    assert int(at_ex1.slider(key="walkthrough_step").value) == step_before + 1

    at_ex1.button(key="walk_prev").click().run()
    assert not at_ex1.exception
    assert int(at_ex1.slider(key="walkthrough_step").value) == step_before

    # Jump to Grow and Shrink events, then verify clamping when max_evals decreases
    at_ex1.button(key="walk_first_grow").click().run()
    assert not at_ex1.exception
    at_ex1.button(key="walk_first_shrink").click().run()
    assert not at_ex1.exception

    at_ex1.selectbox(key="walkthrough_algo").select("Simple Hill Climbing").run()
    assert not at_ex1.exception

    at_ex1.toggle(key="walkthrough_zoom").set_value(True).run()
    assert not at_ex1.exception
    at_ex1.toggle(key="walkthrough_fog").set_value(True).run()
    assert not at_ex1.exception

    # Open modal Pop-up for Eval #2, switch inside modal to Eval #3 and #1, then close
    at_ex1.selectbox(key="walkthrough_algo").select("Adaptive Momentum Hill Climbing").run()
    assert not at_ex1.exception
    at_ex1.checkbox(key="walk_show_all_rows").set_value(True).run()
    assert not at_ex1.exception
    at_ex1.number_input(key="walk_popup_picker").set_value(2).run()
    at_ex1.button(key="walk_open_picked_popup").click().run()
    assert not at_ex1.exception
    assert int(at_ex1.selectbox(key="walk_popup_eval").value) == 2
    at_ex1.button(key="walk_pop_next").click().run()
    assert not at_ex1.exception
    assert int(at_ex1.selectbox(key="walk_popup_eval").value) == 3
    at_ex1.selectbox(key="walk_popup_eval").select(1).run()
    assert not at_ex1.exception
    at_ex1.button(key="walk_pop_close").click().run()
    assert not at_ex1.exception

    # Exercise sequential range clamping in multi-eval browser
    at_ex1.number_input(key="walk_multi_start").set_value(5).run()
    assert not at_ex1.exception
    assert int(at_ex1.number_input(key="walk_multi_end").value) >= 5

    # Push walkthrough_step high, then lower sidebar max_evals -> must clamp cleanly without error
    at_ex1.slider(key="walkthrough_step").set_value(120).run()
    assert not at_ex1.exception
    max_eval_slider = next(s for s in at_ex1.sidebar.slider if s.label == "Max objective evaluations")
    max_eval_slider.set_value(20).run()
    assert not at_ex1.exception
    assert int(at_ex1.slider(key="walkthrough_step").value) <= 20

    at_ex1.sidebar.selectbox[0].select("sinusoidal").run()
    assert not at_ex1.exception
