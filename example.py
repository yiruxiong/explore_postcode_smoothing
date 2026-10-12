"""
example.py – Motor insurance postcode smoothing demo
=====================================================
This script walks through a complete end-to-end workflow for a motor
insurance pricing analyst who wants to:

    1. Train a simple base model.
    2. Compute geographic residuals.
    3. Use H3-based postcode smoothing to remove noise from the residuals.
    4. Produce a geographically adjusted premium.

We use synthetic data so the script runs without any real policy data.
Replace the ``df`` construction block with your own data loading.
"""

import numpy as np
import pandas as pd

from postcode_smoother import PostcodeSmoother
from postcode_smoother.residuals import (
    apply_smoothed_residual,
    compute_multiplicative_residuals,
    summarise_residuals_by_hex,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Synthetic dataset
# ─────────────────────────────────────────────────────────────────────────────
# We create 200 fake policies spread across a handful of UK postal districts.
# In a real project you would load a CSV / database table here.

np.random.seed(42)

# A selection of real UK postcode districts (the first part of the postcode).
postcode_districts = [
    "SW1A 2AA",  # Westminster
    "EC2M 6SQ",  # City of London
    "M1 1AD",    # Manchester city centre
    "B1 1BB",    # Birmingham
    "LS1 4AP",   # Leeds
    "G1 1AB",    # Glasgow
    "E1 6RF",    # East London
    "W1A 1AA",   # West End
    "N1 9GU",    # Islington
    "SE1 7PB",   # Southwark
]

n_policies = 500

# Assign a random postcode to each policy.
postcodes = np.random.choice(postcode_districts, size=n_policies)

# Simulate a base premium (e.g. from a GLM) – varies by postcode area.
base_map = {
    "SW1A 2AA": 450,
    "EC2M 6SQ": 380,
    "M1 1AD":   310,
    "B1 1BB":   290,
    "LS1 4AP":  270,
    "G1 1AB":   320,
    "E1 6RF":   420,
    "W1A 1AA":  460,
    "N1 9GU":   400,
    "SE1 7PB":  390,
}

base_premiums = np.array([base_map[p] for p in postcodes], dtype=float)

# Simulate observed claim frequencies with some geographic variation + noise.
# True underlying multiplier per area – these represent the "true" geographic
# effects that we want the smoother to recover.
geo_effect_map = {
    "SW1A 2AA": 1.15,
    "EC2M 6SQ": 0.90,
    "M1 1AD":   1.05,
    "B1 1BB":   0.95,
    "LS1 4AP":  1.10,
    "G1 1AB":   0.85,
    "E1 6RF":   1.20,
    "W1A 1AA":  1.10,
    "N1 9GU":   1.05,
    "SE1 7PB":  0.98,
}

true_geo_effects = np.array([geo_effect_map[p] for p in postcodes])

# Observed claim frequency = base * true geo effect + noise.
noise = np.random.normal(0, 0.15, size=n_policies)
observed_claim_freq = (base_premiums / 400) * true_geo_effects + noise
# Clip to ensure non-negative values (claim frequencies can't be negative).
observed_claim_freq = observed_claim_freq.clip(min=0.0)

# GLM-fitted claim frequency (pretend the model only partially captured the
# geographic effect – it used the grand mean of the geo effects).
fitted_claim_freq = (base_premiums / 400) * np.random.uniform(0.9, 1.1, n_policies)
fitted_claim_freq = fitted_claim_freq.clip(min=0.01)

# Exposure (earned car years) – some policies have more weight than others.
exposure = np.random.uniform(0.5, 2.0, size=n_policies)

df = pd.DataFrame(
    {
        "policy_id":   range(n_policies),
        "postcode":    postcodes,
        "base_premium": base_premiums,
        "observed_claim_freq": observed_claim_freq,
        "fitted_claim_freq":   fitted_claim_freq,
        "exposure":    exposure,
    }
)

print(f"Dataset: {len(df)} policies across {df['postcode'].nunique()} postcode areas\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Compute multiplicative geographic residuals
# ─────────────────────────────────────────────────────────────────────────────
# The residual is how much the observed claim frequency deviates from what
# the GLM predicted.  A residual of 1.2 means the area is 20 % riskier than
# the model said.

df["residual"] = compute_multiplicative_residuals(
    df["observed_claim_freq"],
    df["fitted_claim_freq"],
)

print("Raw residuals (first 5 rows):")
print(df[["postcode", "observed_claim_freq", "fitted_claim_freq", "residual"]].head())
print(f"\nMean raw residual: {df['residual'].mean():.4f}")
print(f"Std  raw residual: {df['residual'].std():.4f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Smooth the residuals with PostcodeSmoother
# ─────────────────────────────────────────────────────────────────────────────
# We fit the smoother on the training set residuals, asking it to:
#   • Use H3 resolution 7 (~5 km² per hex) for spatial granularity.
#   • Include 2 rings of neighbours (k=2 → up to 19 hexagons per point).
#   • Weight closer hexes more heavily (inverse_distance).
#   • Fall back to global mean if neighbourhood exposure < 2 car-years.

smoother = PostcodeSmoother(
    h3_resolution=7,
    k_ring=2,
    weight_scheme="inverse_distance",
    min_exposure=2.0,
)

# fit_transform is convenient for training-set inspection; in production you
# would fit() on training data and transform() on unseen data.
df["smoothed_residual"] = smoother.fit_transform(
    df,
    postcode_col="postcode",
    value_col="residual",
    exposure_col="exposure",
)

print("Smoothed residuals (first 5 rows):")
print(df[["postcode", "residual", "smoothed_residual"]].head())
print(f"\nMean smoothed residual: {df['smoothed_residual'].mean():.4f}")
print(f"Std  smoothed residual: {df['smoothed_residual'].std():.4f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  Apply smoothed residual to the base premium
# ─────────────────────────────────────────────────────────────────────────────
# The adjusted premium = base premium × smoothed geographic factor.

df["adjusted_premium"] = apply_smoothed_residual(
    df["base_premium"],
    df["smoothed_residual"],
    residual_type="multiplicative",
)

print("Premium comparison (first 10 rows):")
print(
    df[["postcode", "base_premium", "smoothed_residual", "adjusted_premium"]]
    .head(10)
    .to_string(index=False)
)

# ─────────────────────────────────────────────────────────────────────────────
# 5.  Hex-level diagnostics
# ─────────────────────────────────────────────────────────────────────────────
# Summarise residuals per H3 hex to understand which geographic areas the
# model is over- or under-pricing.

hex_df = smoother.get_hex_dataframe()
print(f"\nNumber of H3 hex cells (resolution {smoother.h3_resolution}): {len(hex_df)}")
print("Hex summary (first 5 rows):")
print(hex_df.head())

# ─────────────────────────────────────────────────────────────────────────────
# 6.  Summary statistics by postcode area
# ─────────────────────────────────────────────────────────────────────────────
summary = (
    df.groupby("postcode")
    .agg(
        count=("policy_id", "count"),
        mean_base_premium=("base_premium", "mean"),
        mean_smoothed_residual=("smoothed_residual", "mean"),
        mean_adjusted_premium=("adjusted_premium", "mean"),
    )
    .round(2)
)

print("\nPostcode area summary:")
print(summary.to_string())
print("\nDone.")
