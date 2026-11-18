"""Exercise 2: Genetic Algorithm Variations — interactive image reconstruction."""

from __future__ import annotations

from dataclasses import asdict, replace
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
import plotly.graph_objects as go
import streamlit as st

from bio_inspired_optimisation.genetic_algorithm import GAConfig, GAResult, run_genetic_algorithm


st.set_page_config(page_title="Exercise 2: Genetic Algorithms", page_icon="🧬", layout="wide")
st.title("Exercise 2: Genetic Algorithm Variations")
st.caption("Population-based optimization · Reconstructing a grayscale image · 16×16 pixels and beyond")

ASSETS = Path(__file__).resolve().parents[1] / "assets"


def make_grayscale_target(image: Image.Image, resolution: int) -> np.ndarray:
    """Composite transparent backgrounds onto white and resize without distortion."""
    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    grayscale = Image.alpha_composite(white, rgba).convert("L")
    square = ImageOps.pad(
        grayscale,
        (resolution, resolution),
        method=Image.Resampling.LANCZOS,
        color=255,
    )
    return np.asarray(square, dtype=np.uint8).copy()


def pixel_preview(image: np.ndarray, size: int = 320) -> Image.Image:
    """Nearest-neighbor enlargement so the actual pixels remain visible."""
    return Image.fromarray(image, mode="L").resize((size, size), Image.Resampling.NEAREST)


def as_png_bytes(image: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(image, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def as_gif_bytes(target: np.ndarray, result: GAResult) -> bytes:
    """Animated left/right comparison, with no need for another optimization run."""
    count = len(result.best_frames)
    sampled = np.unique(np.linspace(0, count - 1, min(count, 28)).astype(int))
    frames = []
    target_img = pixel_preview(target, 240)
    for index in sampled:
        frame = Image.new("RGB", (504, 292), (245, 246, 249))
        frame.paste(target_img, (8, 34))
        frame.paste(pixel_preview(result.best_frames[int(index)], 240), (256, 34))
        draw = ImageDraw.Draw(frame)
        draw.text((12, 10), "Target", fill=(24, 34, 48))
        draw.text((258, 10), f"Generation {index}", fill=(24, 34, 48))
        frames.append(frame)
    buf = BytesIO()
    frames[0].save(
        buf, format="GIF", save_all=True, append_images=frames[1:],
        duration=[140] * (len(frames) - 1) + [1150], loop=0,
        optimize=False, disposal=2,
    )
    return buf.getvalue()


def fitness_chart(
    result: GAResult,
    name: str | None = None,
    x_axis: str = "evaluations",
) -> go.Figure:
    h = result.history
    use_generations = x_axis == "generations"
    x = [r.generation if use_generations else r.eval_count for r in h]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=[r.best_mse for r in h], mode="lines", name="Best so far",
        line=dict(color="#2d8bd1", width=3),
        customdata=[r.generation for r in h],
        hovertemplate="Evals: %{x}<br>Generation: %{customdata}<br>Best MSE: %{y:.6f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=[r.mean_mse for r in h], mode="lines", name="Population mean",
        line=dict(color="#ef9b39", width=2),
        customdata=[r.generation for r in h],
        hovertemplate="Evals: %{x}<br>Generation: %{customdata}<br>Mean MSE: %{y:.6f}<extra></extra>",
    ))
    if result.config.target_mse is not None:
        fig.add_hline(y=result.config.target_mse, line_dash="dot", line_color="#3fa57a",
                      annotation_text="Target MSE")
    fig.update_layout(
        title=name or "Fitness improvement over objective evaluations",
        xaxis_title=("Generation (generation 0 = initial population)"
                     if use_generations else
                     "Objective evaluations (including initial population)"),
        yaxis_title="Normalized MSE ↓ (lower is better)",
        hovermode="x unified", legend=dict(orientation="h", y=1.16),
        margin=dict(t=90, b=30, l=20, r=20), height=380,
    )
    return fig


def diversity_chart(result: GAResult) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[r.generation for r in result.history],
        y=[r.diversity for r in result.history],
        mode="lines", name="Diversity",
        line=dict(color="#a974cf", width=2.5),
        hovertemplate="Generation %{x}<br>Pixel diversity %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Population diversity over generations", xaxis_title="Generation",
        yaxis_title="Mean per-pixel standard deviation / 255",
        margin=dict(t=60, b=30, l=20, r=20), height=285,
    )
    return fig


def experiment_pairs(which: str, config: GAConfig) -> list[tuple[str, GAConfig]]:
    """Two controlled algorithm variants, keeping all other settings unchanged."""
    if which == "Crossover":
        return [("Uniform pixels", replace(config, crossover="uniform")),
                ("Spatial rectangle", replace(config, crossover="rectangle"))]
    if which == "Mutation operator":
        return [("Gaussian noise", replace(config, mutation="gaussian")),
                ("Random pixel reset", replace(config, mutation="reset"))]
    if which == "Mutation rate":
        return [("2% per pixel", replace(config, mutation_rate=0.02)),
                ("18% per pixel", replace(config, mutation_rate=0.18))]
    if which == "Initialization":
        return [("Fully random", replace(config, initialization="random")),
                ("Mean-tone informed", replace(config, initialization="mean_tone"))]
    if which == "Parent selection":
        return [("Tournament", replace(config, parent_selection="tournament")),
                ("Rank-based", replace(config, parent_selection="rank"))]
    return [("Elitism / turnover", replace(config, survivor_selection="elitism")),
            ("(μ+λ) best survivors", replace(config, survivor_selection="plus"))]


# Shared sidebar for each documentation tab.
with st.sidebar:
    st.header("GA Experiment Settings")
    sample_choice = st.selectbox(
        "Target image",
        ["Simple cat", "Upload your own image"],
    )
    uploaded_file = st.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg", "bmp", "webp"],
        help="Any RGB/RGBA image is converted to grayscale; transparency is filled with white.",
    ) if sample_choice == "Upload your own image" else None
    resolution = st.select_slider(
        "Resolution (width × height)", options=[16, 24, 32, 48, 64], value=16,
        help="16×16 = 256 genes; 64×64 = 4,096 genes. Higher resolution is more expensive.",
    )

    if uploaded_file is not None:
        try:
            source = Image.open(BytesIO(uploaded_file.getvalue()))
            source.load()
        except Exception as exc:
            st.error(f"The uploaded file cannot be opened: {exc}")
            source = Image.open(ASSETS / "cat.jpg")
    elif sample_choice == "Hill Climb artwork (sample)":
        source = Image.open(ASSETS / "cat.jpg")
    else:
        source = Image.open(ASSETS / "cat.jpg")
        if sample_choice == "Upload your own image":
            st.info("No file selected yet. Using the sample cat for now.")
    target = make_grayscale_target(source, resolution)
    st.image(pixel_preview(target, 180), caption=f"Target preview · {resolution} × {resolution}", width=180)

    with st.expander("Population & Termination", expanded=True):
        population = st.slider("Population size (μ)", 10, 140, 48, step=2)
        generations = st.slider("Maximum generations", 20, 500, 120, step=10)
        seed = int(st.number_input("Random seed", min_value=0, max_value=2_000_000, value=123))
        use_target = st.checkbox("Stop when a target MSE is reached", value=False)
        target_threshold = st.number_input(
            "Target normalized MSE", min_value=0.0, max_value=1.0,
            value=0.005, format="%.4f", step=0.001,
            disabled=not use_target,
        )

    with st.expander("Genetic Operators", expanded=True):
        initialization = st.selectbox("Initialization", ["hybrid", "random", "mean_tone"],
                                       format_func={"hybrid": "Hybrid: random + mean-tone", "random": "Fully random", "mean_tone": "Mean-tone + noise"}.get)
        parent_selection = st.selectbox("Parent selection", ["tournament", "rank"],
                                        format_func={"tournament": "Tournament", "rank": "Rank-based"}.get)
        tournament_size = st.slider("Tournament size", 2, min(8, population), 3,
                                    disabled=parent_selection != "tournament")
        crossover = st.selectbox("Crossover", ["uniform", "rectangle", "two_point"],
                                  format_func={"uniform": "Uniform pixels", "rectangle": "Spatial rectangle", "two_point": "Two-point"}.get)
        crossover_rate = st.slider("Crossover probability", 0.0, 1.0, 0.90, step=0.05)
        mutation = st.selectbox("Mutation", ["gaussian", "reset"],
                                 format_func={"gaussian": "Gaussian pixel noise", "reset": "Random pixel reset"}.get)
        mutation_rate = st.slider("Mutation probability (per pixel)", 0.0, 0.50, 0.06, step=0.01)
        mutation_strength = st.slider("Gaussian mutation strength (0–255)", 1.0, 100.0, 35.0,
                                      step=1.0, disabled=mutation != "gaussian")
        survivor_selection = st.selectbox("Survivor selection", ["elitism", "plus"],
                                          format_func={"elitism": "Generational + elitism", "plus": "(μ+λ) best survivors"}.get)
        elites = st.slider("Elites retained", 1, min(12, population - 1), 2,
                           disabled=survivor_selection != "elitism")

    st.caption("The GA evaluates only newly created offspring; retained parents are not re-evaluated.")

config = GAConfig(
    population_size=population, generations=generations,
    initialization=initialization, parent_selection=parent_selection,
    tournament_size=tournament_size, crossover=crossover, crossover_rate=crossover_rate,
    mutation=mutation, mutation_rate=mutation_rate, mutation_strength=mutation_strength,
    survivor_selection=survivor_selection, elites=elites,
    target_mse=float(target_threshold) if use_target else None, seed=seed,
)
settings_id = (sha256(target.tobytes()).hexdigest(), config)

tab_intro, tab_methods, tab_results, tab_discussion = st.tabs(
    ["1. Introduction", "2. Methods", "3. Results", "4. Discussion"]
)

with tab_intro:
    st.header("1. Introduction")
    st.markdown(r"""
A **Genetic Algorithm (GA)** is a population-based evolutionary optimization method.
Instead of improving one single solution (as in Hill Climbing), it evolves **many candidates**
through selection, crossover, mutation, and survivor selection.

**For this exercise:** Each individual is a complete grayscale picture. The goal is to reproduce
a target image, starting with **16 × 16 pixels = 256 decision variables**.

**Basic cycle**
1. Create a population of candidate images.
2. Evaluate how similar each image is to the target.
3. Select promising parents and combine their pixels (crossover).
4. Randomly modify some pixels (mutation).
5. Select survivors and repeat for more generations.

**Strengths:** Population diversity, flexible representations and genetic operators, no gradients required.
**Weaknesses:** Many fitness evaluations, parameter sensitivity, possible premature convergence.
""")

with tab_methods:
    st.header("2. Methods")
    st.markdown(r"""
**Representation — One grayscale intensity per gene**
- Chromosome: a flattened $H\times W$ image stored as `uint8` values from **0 (black)** to **255 (white)**.
- Population: a 2D array of shape `(population_size, H * W)`.
- Unlike binary coding, this representation directly matches the image pixels and needs no decoding.

**Fitness — Normalized Mean Squared Error (minimize)**

$$
\operatorname{MSE}_{norm}(I,T)=\frac{1}{HW}\sum_{p=1}^{HW}
\left(\frac{I_p-T_p}{255}\right)^2
$$

- **0** means an exact match; a lower value is better.
- Dividing by 255 and averaging pixels makes MSE comparable across image resolutions.
- Each evaluated individual counts as **one fitness function evaluation**.

**Initialization**
- **Random:** Every gene is sampled uniformly from 0–255.
- **Mean-tone + noise:** Only the target's *global mean brightness* is used to seed noisy candidates; no target pixel positions are copied.
- **Hybrid:** Half random, half mean-tone — combines diversity with a brightness-informed starting point.

**Parent selection**
- **Tournament:** Choose the best of $k$ randomly sampled candidates. Higher $k$ means stronger selection pressure.
- **Rank-based:** Assign selection probabilities according to sorted fitness, rather than raw MSE.

**Crossover**
- **Uniform:** Independently choose each pixel from either parent.
- **Two-point:** Inherit a contiguous segment of the flattened chromosome from the other parent.
- **Spatial rectangle:** Inherit an entire rectangular image patch to preserve local structure.

**Mutation**
- **Gaussian:** Perturb selected pixel intensities with zero-mean Gaussian noise and clip to $[0,255]$.
- **Random reset:** Replace selected pixel intensities with new random values.
- The mutation probability applies **independently to each pixel**, not once per individual.

**Survivor selection (innovation)**
- **Generational + elitism:** Keep the top $e$ parents, replace everyone else with new offspring.
- **$(\mu+\lambda)$ selection:** Generate $\lambda=\mu$ offspring, then keep the best $\mu$ candidates from parents and children.
- Elitism protects the best solution; strong survivor pressure can reduce diversity.

**Termination and exact evaluation counting**
- Stop after the configured maximum number of generations or when the optional MSE threshold is met.
- Initial population: $P$ evaluations, counted as **generation 0**.
- Each generation: **$P-e$** new evaluations with elitism, or **$P$** with $(\mu+\lambda)$ selection.
- Existing parents and retained elites are **not evaluated again**.
- Store the best-so-far individual, best MSE, population mean, diversity, and evaluation count after each generation.
""")

with tab_results:
    st.header("3. Results — Grayscale Image Reconstruction")
    action, notice = st.columns([1, 2])
    with action:
        rerun = st.button("▶ Run genetic algorithm", type="primary", width="stretch")
    with notice:
        st.caption(f"Current experiment: {resolution}×{resolution} = {resolution**2:,} genes · "
                   f"{population} individuals · up to {generations} generations · seed {seed}")

    if "ga_main_run" not in st.session_state or rerun:
        with st.spinner("Evolving image population and recording every generation..."):
            st.session_state["ga_main_run"] = (
                settings_id,
                target.copy(),
                run_genetic_algorithm(target, config),
            )

    old_id, shown_target, result = st.session_state["ga_main_run"]
    if old_id != settings_id:
        st.warning("You changed the settings. The results below still belong to the **previous run**. "
                   "Click **Run genetic algorithm** to update them.")

    last = result.history[-1]
    improvement = 100 * (result.initial_best_mse - result.best_mse) / max(result.initial_best_mse, 1e-12)
    cols = st.columns(5)
    cols[0].metric("Best normalized MSE ↓", f"{result.best_mse:.6f}")
    cols[1].metric("Improvement from Gen 0", f"{improvement:.1f}%")
    cols[2].metric("Generations completed", f"{last.generation:,}")
    cols[3].metric("Fitness evaluations", f"{result.eval_count:,}")
    cols[4].metric("Runtime", f"{result.runtime_seconds:.2f} s")
    st.caption(f"Stopped because: **{result.stop_reason}**. "
               "MSE is measured against the **target image used in this run**.")

    st.subheader("Reference vs. reconstructed image")
    a, b, c = st.columns(3)
    with a:
        st.image(pixel_preview(shown_target), caption="Target (ground truth)", width="stretch")
    with b:
        st.image(pixel_preview(result.best_image), caption=f"Best individual · MSE {result.best_mse:.6f}", width="stretch")
    with c:
        difference = np.abs(result.best_image.astype(np.int16) - shown_target.astype(np.int16)).astype(np.uint8)
        st.image(pixel_preview(difference), caption="Absolute pixel error · brighter = larger error", width="stretch")

    st.subheader("Fitness and diversity over time")
    view_axis = st.radio(
        "Convergence x-axis", ["evaluations", "generations"],
        format_func={"evaluations": "Objective evaluations", "generations": "Generations"}.get,
        horizontal=True,
        help="Generations show learning steps; evaluations measure how many image fitness calculations were used.",
    )
    st.plotly_chart(fitness_chart(result, x_axis=view_axis), width="stretch", config={"displaylogo": False})
    st.plotly_chart(diversity_chart(result), width="stretch", config={"displaylogo": False})
    st.caption("Fitness curves show **best-so-far** (non-increasing with elitism) and **current population mean**. "
               "Diversity measures the average population spread in pixel intensities. "
               "A low diversity value can indicate premature convergence.")

    st.subheader("How the best image improves")
    # Visual milestones are useful even before interacting with the replay slider.
    milestone_indices = sorted(set([0, len(result.best_frames) // 4,
                                    len(result.best_frames) // 2,
                                    len(result.best_frames) - 1]))
    milestone_cols = st.columns(len(milestone_indices))
    for generation_index, col in zip(milestone_indices, milestone_cols):
        with col:
            st.image(pixel_preview(result.best_frames[generation_index], 240),
                     caption=(f"Generation {generation_index} · "
                              f"MSE {result.history[generation_index].best_mse:.4f}"),
                     width="stretch")

    st.subheader("Replay evolution generation by generation")
    selected_gen = st.slider("Show best image after generation", 0, len(result.best_frames) - 1,
                             value=len(result.best_frames) - 1, key="ga_generation_replay")
    replay_image = result.best_frames[selected_gen]
    replay_record = result.history[selected_gen]
    replay_a, replay_b = st.columns([1, 2])
    with replay_a:
        st.image(pixel_preview(replay_image), caption=f"Best so far · generation {selected_gen}", width="stretch")
    with replay_b:
        st.metric("Best MSE at this generation", f"{replay_record.best_mse:.6f}")
        st.metric("Evaluations up to this generation", f"{replay_record.eval_count:,}")
        st.metric("Population diversity", f"{replay_record.diversity:.3f}")

with tab_discussion:
    st.header("4. Discussion")
    st.markdown(r"""
**What should we expect?**
- The best-so-far MSE should not increase when keeping elites or the best parents.
- Strong parent/survivor selection may improve fitness quickly but reduce population diversity.
- Mutation helps create new pixel values; too much mutation can destroy useful combinations.
- At larger image resolutions, the number of genes grows quadratically: $16^2=256$ vs. $64^2=4096$.

**Exploration vs. exploitation**
- **Exploration:** Random initialization, higher mutation rates, weaker selection pressure.
- **Exploitation:** Tournament pressure, low mutation rates, keeping strong existing individuals.
- **Uniform crossover** mixes individual pixels; **rectangle crossover** preserves spatial patches.
- The most effective setting is *problem-dependent* — compare strategies experimentally rather than assuming one always wins.

**Important interpretation of the Results plots**
- **Best MSE** measures the strongest image seen so far, not the average individual.
- **Population mean MSE** shows overall population quality.
- **Diversity** shows how much the population's pixel intensities differ.
- **Fitness evaluations** are a fairer cost measure than generations alone because survivor strategies can evaluate different numbers of offspring.

**Limitations / possible improvements**
- A separate gene for each pixel ignores image structure and scales poorly to high resolutions.
- Pixel-wise MSE can rate a slightly shifted object poorly even if it looks similar.
- Better choices might include block-based representations, multi-resolution optimization,
  adaptive mutation, or a perceptual metric such as SSIM.
- Mean-tone initialization only uses the **global average intensity** of the target;
  it does not copy the target itself.

**Use the A/B comparison** in Results to support claims about operator choices with actual measurements.
For a stronger conclusion, repeat comparisons with different seeds and images.
""")
