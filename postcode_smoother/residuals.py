"""
residuals.py – Residual modelling utilities
============================================
In motor insurance pricing, a Generalised Linear Model (GLM) is trained on
many rating factors (age, vehicle group, region, …).  After fitting the GLM,
we look at the *residuals* – the difference (or ratio) between what the model
predicted and what actually happened.

Spatial smoothing then asks: "Is there a geographic pattern in the residuals
that the model missed?"  If nearby postcodes consistently have residuals in
the same direction, the model may be systematically mispricing those areas.

This module provides helpers to:
    1. Compute raw multiplicative geographic residuals from observed vs.
       fitted values.
    2. Hand those residuals to the PostcodeSmoother so the H3-based spatial
       average removes the noise.
    3. Merge the smoothed residual back onto the original data for re-scoring.
"""

import numpy as np
import pandas as pd


def compute_multiplicative_residuals(
    observed: pd.Series,
    fitted: pd.Series,
    epsilon: float = 1e-6,
) -> pd.Series:
    """
    Compute the multiplicative (ratio) residual for each row.

    Formula:
        residual = observed / fitted

    A residual of 1.0 means the model is spot-on.
    A residual of 1.2 means the area is 20 % worse than predicted.
    A residual of 0.8 means the area is 20 % better than predicted.

    Parameters
    ----------
    observed : pd.Series
        Actual claim frequency (or cost per unit) per row.
    fitted : pd.Series
        GLM-predicted claim frequency (or cost per unit) per row.
    epsilon : float
        Small floor applied to ``fitted`` to avoid division by zero.

    Returns
    -------
    pd.Series
        Multiplicative residual for each row, shape matches inputs.
    """
    # Guard against division by zero for rows with very small predictions.
    safe_fitted = fitted.clip(lower=epsilon)
    return observed / safe_fitted


def compute_additive_residuals(
    observed: pd.Series,
    fitted: pd.Series,
) -> pd.Series:
    """
    Compute the additive (difference) residual for each row.

    Formula:
        residual = observed - fitted

    Positive values mean the model under-predicted; negative means over-predicted.

    Parameters
    ----------
    observed : pd.Series
        Actual values.
    fitted : pd.Series
        Model-predicted values.

    Returns
    -------
    pd.Series
        Additive residual for each row.
    """
    return observed - fitted


def apply_smoothed_residual(
    base_premium: pd.Series,
    smoothed_residual: pd.Series,
    residual_type: str = "multiplicative",
) -> pd.Series:
    """
    Apply the geographically smoothed residual factor to a base premium.

    For multiplicative residuals:
        adjusted_premium = base_premium * smoothed_residual

    For additive residuals:
        adjusted_premium = base_premium + smoothed_residual

    Parameters
    ----------
    base_premium : pd.Series
        Base technical premium before geographic adjustment.
    smoothed_residual : pd.Series
        Smoothed geographic residual (output of PostcodeSmoother.transform).
    residual_type : str
        ``"multiplicative"`` (default) or ``"additive"``.

    Returns
    -------
    pd.Series
        Geographically adjusted premium.
    """
    if residual_type == "multiplicative":
        return base_premium * smoothed_residual
    elif residual_type == "additive":
        return base_premium + smoothed_residual
    else:
        raise ValueError(
            f"residual_type must be 'multiplicative' or 'additive', got '{residual_type}'"
        )


def summarise_residuals_by_hex(
    df: pd.DataFrame,
    hex_col: str = "h3_cell",
    residual_col: str = "residual",
    weight_col: str | None = None,
) -> pd.DataFrame:
    """
    Aggregate raw residuals to the H3 hex level.

    Useful for diagnostics: plot the hex-level mean residual on a map to see
    which geographic areas the model is over- or under-pricing.

    Parameters
    ----------
    df : pd.DataFrame
        Row-level data containing hex IDs and residuals.
    hex_col : str
        Column name of the H3 hex cell ID.
    residual_col : str
        Column name of the raw residual to aggregate.
    weight_col : str | None
        Optional column of exposure weights (e.g. earned car years).
        If provided, a weighted mean is computed instead of a simple mean.

    Returns
    -------
    pd.DataFrame
        One row per H3 hex with columns ``[hex_col, "mean_residual", "count"]``.
    """
    if weight_col is not None and weight_col in df.columns:
        # Weighted mean: sum(residual * weight) / sum(weight)
        def weighted_mean(group):
            w = group[weight_col]
            r = group[residual_col]
            total_weight = w.sum()
            if total_weight == 0:
                return np.nan
            return (r * w).sum() / total_weight

        agg = (
            df.groupby(hex_col)
            .apply(weighted_mean)
            .reset_index()
        )
        agg.columns = [hex_col, "mean_residual"]
        counts = (
            df.groupby(hex_col)[residual_col]
            .count()
            .reset_index(name="count")
        )
        agg = agg.merge(counts, on=hex_col)
    else:
        agg = (
            df.groupby(hex_col)[residual_col]
            .agg(mean_residual="mean", count="count")
            .reset_index()
        )

    return agg
