# postcode-smoother

A lightweight Python package for **UK postcode smoothing** powered by
[Uber's H3 hexagonal grid](https://www.uber.com/us/en/blog/h3/).

Designed for **motor insurance risk premium modelling**: instead of treating
each postcode in isolation, borrow signal from spatially adjacent postcodes by
averaging observed values (e.g. claim frequency, GLM residuals) across the
H3 hex cells that surround a given location.

---

## Why postcode smoothing?

Motor insurance premiums are partly determined by *where* the policyholder
lives.  A naive approach assigns each postcode its own empirical loss rate, but
this is noisy – especially for rural or low-density postcodes with few
policies.  **Spatial smoothing** solves this by blending a postcode's own
observed rate with those of its geographic neighbours, producing stable
estimates without discarding the real underlying geographic pattern.

---

## Why H3?

[H3](https://h3geo.org/) partitions the Earth into a hierarchical grid of
regular hexagons.  Key advantages:

- **Equal area** – every hexagon at the same resolution has approximately the
  same area, so comparisons across the grid are fair.
- **Neighbour distance uniformity** – all 6 immediate neighbours of a hexagon
  are the same distance from the centre, unlike rectangular grids.
- **Multiple resolutions** – pick the right geographic granularity for your
  data volume (see table below).

| H3 resolution | Approx. area | UK context |
|---|---|---|
| 5 | ~252 km² | County level – very heavy smoothing |
| 6 | ~36 km²  | District level |
| **7** | **~5 km²**  | **Sector level ← recommended default** |
| 8 | ~0.7 km² | Postcode unit level – light smoothing |

---

## Installation

```bash
pip install -r requirements.txt
pip install -e .   # install the package in development mode
```

---

## Quick start

```python
import pandas as pd
from postcode_smoother import PostcodeSmoother
from postcode_smoother.residuals import (
    compute_multiplicative_residuals,
    apply_smoothed_residual,
)

# Your policy-level data
df = pd.read_csv("policies.csv")
# Required columns: postcode, observed_claim_freq, fitted_claim_freq,
#                   exposure, base_premium

# Step 1 – compute geographic residuals from your GLM output
df["residual"] = compute_multiplicative_residuals(
    df["observed_claim_freq"],
    df["fitted_claim_freq"],
)

# Step 2 – smooth the residuals spatially
smoother = PostcodeSmoother(
    h3_resolution=7,          # ~5 km² per hex
    k_ring=2,                  # include 2 rings of neighbours
    weight_scheme="inverse_distance",
    min_exposure=2.0,          # sparse-area fallback threshold
)
df["smoothed_residual"] = smoother.fit_transform(
    df,
    postcode_col="postcode",
    value_col="residual",
    exposure_col="exposure",
)

# Step 3 – apply geographic adjustment to base premium
df["adjusted_premium"] = apply_smoothed_residual(
    df["base_premium"],
    df["smoothed_residual"],
    residual_type="multiplicative",
)
```

Run the full walkthrough:

```bash
python example.py
```

---

## API reference

### `PostcodeSmoother(h3_resolution, k_ring, weight_scheme, min_exposure)`

| Parameter | Default | Description |
|---|---|---|
| `h3_resolution` | `7` | H3 grid resolution (0–15). Higher = smaller hexagons = lighter smoothing. |
| `k_ring` | `1` | Number of neighbour rings to include. k=1 → 7 hexagons, k=2 → 19, k=3 → 37. |
| `weight_scheme` | `"uniform"` | `"uniform"` or `"inverse_distance"` (closer hexes weighted more). |
| `min_exposure` | `0.0` | Minimum neighbourhood exposure to trust local estimates; falls back to global mean if below threshold. |

#### Methods

- **`fit(df, postcode_col, value_col, exposure_col)`** – aggregate values to hex level.
- **`transform(df, postcode_col)`** – return smoothed values for new data.
- **`fit_transform(df, ...)`** – convenience wrapper.
- **`get_hex_dataframe()`** – return fitted hex grid as a DataFrame (useful for mapping).

### `postcode_smoother.residuals`

- **`compute_multiplicative_residuals(observed, fitted)`** – ratio residual.
- **`compute_additive_residuals(observed, fitted)`** – difference residual.
- **`apply_smoothed_residual(base_premium, smoothed_residual, residual_type)`** – apply to premium.
- **`summarise_residuals_by_hex(df, hex_col, residual_col, weight_col)`** – hex-level diagnostics.

---

## Running tests

```bash
pip install pytest
pytest tests/ -v
```
