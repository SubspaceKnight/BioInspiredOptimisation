# Bio-Inspired Optimisation

Exercise portfolio and interactive documentation built with **uv** and **Streamlit** (multipage `pages/` layout).

## Setup & Usage

Install dependencies and create the virtual environment:

```bash
uv sync
```

Launch the Streamlit multipage application:

```bash
uv run streamlit run Home.py
```

Run the test suite:

```bash
uv run pytest
```

## Project Structure

- `Home.py` — Course portfolio landing page and benchmark overview.
- `pages/1_Exercise_01_Hill_Climbing.py` — Exercise 1 documentation (**Introduction**, **Methods**, **Results**, **Discussion**) including an interactive lab with **Fog of War** probe-only visualization and evaluation-by-evaluation replay.
- `src/bio_inspired_optimisation/` — Reusable optimization library (`benchmarks.py`, `neighborhoods.py`, `termination.py`, `hill_climbing.py`, `experiments.py`, `plotting.py`).
- `tests/` — Unit and Streamlit integration tests.
