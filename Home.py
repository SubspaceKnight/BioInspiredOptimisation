"""Landing page for the Bio-Inspired Optimisation exercise portfolio."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from bio_inspired_optimisation.benchmarks import BENCHMARKS

st.set_page_config(
    page_title="Bio-Inspired Optimisation | Portfolio",
    page_icon="🧬",
    layout="wide",
)

ROOT = Path(__file__).resolve().parent
HC_PAGE = "pages/1_Exercise_01_Hill_Climbing.py"
GA_PAGE = "pages/2_Exercise_02_Genetic_Algorithms.py"

st.title("🧬 Bio-Inspired Optimisation")
st.caption("FH JOANNEUM · Data Science & Artificial Intelligence · Winter Term 2026/27")
st.markdown(
    """
An interactive portfolio for **derivative-free optimisation**. Explore the theory,
change algorithm parameters, run experiments, and compare the results.

Each exercise includes **Introduction, Methods, Results, and Discussion**.
"""
)

metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Exercises", "2")
metric_b.metric("Optimisation approaches", "Local + Evolutionary")
metric_c.metric("Hill Climbing benchmarks", str(len(BENCHMARKS)))

st.divider()
st.header("Choose an exercise")

left, right = st.columns(2, gap="large")

with left:
    with st.container(border=True):
        st.subheader("⛰️ Exercise 01 · Hill Climbing")
        st.caption("Single-solution search · Continuous optimisation")
        st.markdown(
            """
**Goal:** Find the minimum of different benchmark functions.

- **Algorithms:** Simple HC and Adaptive Momentum HC
- **Functions:** Quadratic, Sinusoidal, and Ackley
- **Explore:** Neighbourhoods, step size, momentum, and termination
- **Results:** Trajectories, convergence, and paired multi-start comparisons
- **Interactive:** Step-by-step dynamics and Fog of War
"""
        )
        if (ROOT / HC_PAGE).is_file():
            st.page_link(HC_PAGE, label="Open Exercise 01", icon="➡️", use_container_width=True)
        else:
            st.warning("Exercise 01 page not found in the pages/ folder.")

with right:
    with st.container(border=True):
        st.subheader("🧬 Exercise 02 · Genetic Algorithms")
        st.caption("Population-based search · Image reconstruction")
        st.markdown(
            """
**Goal:** Reconstruct a grayscale image by evolving candidate images.

- **Representation:** One pixel intensity per gene
- **Fitness:** Normalised mean squared error (MSE)
- **Explore:** Initialisation, parent selection, crossover, and mutation
- **Results:** Image evolution, fitness curves, diversity, and runtime
- **Interactive:** Generation replay and operator comparisons
"""
        )
        if (ROOT / GA_PAGE).is_file():
            st.page_link(GA_PAGE, label="Open Exercise 02", icon="➡️", use_container_width=True)
        else:
            st.warning("Exercise 02 page not found in the pages/ folder.")

st.divider()
st.subheader("How the exercises differ")
st.markdown(
    """
| | Hill Climbing | Genetic Algorithm |
|---|---|---|
| **Solutions at a time** | One current candidate | A population of candidates |
| **New candidates** | Random neighbour | Crossover and mutation |
| **Search guidance** | Improvement and optional momentum | Parent and survivor selection |
| **Main challenge** | Local minima and step-size choice | Diversity versus convergence |
| **Typical output** | Best coordinates and objective value | Best reconstructed image and MSE |

Both approaches are **stochastic, derivative-free** and evaluated using objective function values.
"""
)

with st.expander("Hill Climbing benchmark functions"):
    rows = []
    for spec in BENCHMARKS.values():
        lower, upper = spec.default_bounds
        rows.append(
            {
                "Function": spec.name,
                "Dimensions": spec.default_dim,
                "Default domain (per dimension)": f"[{lower:.3f}, {upper:.3f}]",
                "Global optimum": spec.global_min_description,
                "Landscape": spec.landscape_notes,
            }
        )
    st.dataframe(rows, hide_index=True, use_container_width=True)

st.info(
    "**Recommended next step:** Open **Exercise 02** to test how population size, "
    "mutation, and crossover affect the image reconstruction. "
    "Start at 16 × 16 pixels before increasing the resolution."
)
