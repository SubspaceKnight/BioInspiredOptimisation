"""Main entry page for the Bio-Inspired Optimisation multipage Streamlit app."""

from __future__ import annotations

import streamlit as st

from bio_inspired_optimisation.benchmarks import BENCHMARKS

st.set_page_config(
    page_title="Bio-Inspired Optimisation",
    page_icon="🧬",
    layout="wide",
)

st.title("Bio-Inspired Optimisation — Exercise Portfolio")
st.caption("FH JOANNEUM · Interactive Theory, Implementation & Experimental Analysis")

st.markdown(
    """
Welcome to the **Bio-Inspired Optimisation** interactive documentation and laboratory.
This repository uses `uv` for reproducible package management and Streamlit's `pages/`
multipage architecture so new exercises can be added cleanly throughout the semester.
"""
)

col1, col2, col3 = st.columns(3)
col1.metric("Current Exercise", "Exercise 1: Hill Climbing")
col2.metric("Optimisation Mode", "Minimization (f_new < f_old)")
col3.metric("Benchmarks Implemented", f"{len(BENCHMARKS)} (1D & 2D)")

st.divider()

st.subheader("Exercises")
st.markdown(
    """
- **Exercise 1 — Hill Climbing** *(open from the left sidebar)*
  - **1. Introduction:** Derivative-free local search theory, minimization vs. maximization, zero-order vs. gradient-based methods (Adam/GD), and complexity.
  - **2. Methods:** Reusable solver design (`x0`, `objective_fn`, `neighborhood_fn`, `termination`), exact evaluation counting, and **Simple HC** vs. **Adaptive Momentum HC** (Adam-inspired first/second moments over successful steps).
  - **3. Results:** Interactive trajectory plots (1D line, 2D contour, 3D surface), adjustable step size/neighborhood/bounds, **Fog of War** black-box probe view with step-by-step replay slider, and paired multi-start statistics.
  - **4. Discussion:** Empirical takeaways on convergence speed, precision, local-minimum trapping, parameter sensitivity, and limitations.
  - **5. Step-by-Step Dynamics:** Interactive single-evaluation walkthrough with live substituted equations, search windows/vectors, streaks (`accept_streak` / `reject_streak`), and variable transitions (`drift`, `offset ε`, `m`, `v`, `σ`).
"""
)

st.subheader("Benchmark Suite Overview")
rows = []
for spec in BENCHMARKS.values():
    lo, hi = spec.default_bounds
    rows.append(
        {
            "Benchmark": spec.name,
            "Default Dim": spec.default_dim,
            "Test Domain": f"[{lo:.3f}, {hi:.3f}]",
            "Global Minimum": spec.global_min_description,
            "Landscape Character": spec.landscape_notes,
        }
    )
st.dataframe(rows, hide_index=True, width="stretch")

st.info("Select **1 Exercise 01 Hill Climbing** in the left sidebar to explore the full report and interactive lab.")
