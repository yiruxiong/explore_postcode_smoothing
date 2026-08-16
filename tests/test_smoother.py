"""
tests/test_smoother.py
======================
Unit tests for the postcode_smoother package.

All tests supply pre-computed lat/lon values directly so that no internet
connection (pgeocode download) is required to run the test suite.
"""

import numpy as np
import pandas as pd
import pytest

from postcode_smoother import PostcodeSmoother
from postcode_smoother.residuals import (
    apply_smoothed_residual,
    compute_additive_residuals,
    compute_multiplicative_residuals,
    summarise_residuals_by_hex,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

# Real approximate lat/lon coordinates for a selection of UK postcode areas.
# We supply these directly so tests do not need a pgeocode download.
POSTCODE_COORDS = {
    "SW1A 2AA": (51.5034, -0.1276),  # Westminster
    "EC2M 6SQ": (51.5177, -0.0832),  # City of London
    "M1 1AD":   (53.4808, -2.2426),  # Manchester
    "B1 1BB":   (52.4796, -1.9026),  # Birmingham
    "LS1 4AP":  (53.7996, -1.5491),  # Leeds
}

SAMPLE_POSTCODES = list(POSTCODE_COORDS.keys())


@pytest.fixture
def sample_df():
    """
    A small DataFrame of UK postcodes with pre-computed lat/lon,
    a numeric 'value' column, and an 'exposure' column.
    """
    np.random.seed(0)
    n_per_pc = 4
    postcodes = SAMPLE_POSTCODES * n_per_pc
    n = len(postcodes)
    lats = [POSTCODE_COORDS[p][0] for p in postcodes]
    lons = [POSTCODE_COORDS[p][1] for p in postcodes]
    values = np.random.uniform(0.8, 1.3, size=n)
    exposure = np.random.uniform(0.5, 2.0, size=n)
    return pd.DataFrame(
        {
            "postcode": postcodes,
            "latitude": lats,
            "longitude": lons,
            "value": values,
            "exposure": exposure,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# PostcodeSmoother – basic contract tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPostcodeSmoother:
    def test_fit_returns_self(self, sample_df):
        """fit() should return the smoother instance for method chaining."""
        smoother = PostcodeSmoother()
        result = smoother.fit(sample_df, lat_col="latitude", lon_col="longitude")
        assert result is smoother

    def test_transform_length(self, sample_df):
        """transform() must return a Series with the same length as input."""
        smoother = PostcodeSmoother()
        smoother.fit(sample_df, lat_col="latitude", lon_col="longitude")
        out = smoother.transform(sample_df, lat_col="latitude", lon_col="longitude")
        assert len(out) == len(sample_df)

    def test_transform_index_preserved(self, sample_df):
        """transform() must preserve the DataFrame index."""
        smoother = PostcodeSmoother()
        smoother.fit(sample_df, lat_col="latitude", lon_col="longitude")
        out = smoother.transform(sample_df, lat_col="latitude", lon_col="longitude")
        pd.testing.assert_index_equal(out.index, sample_df.index)

    def test_fit_transform_matches(self, sample_df):
        """fit_transform() must give the same result as fit() + transform()."""
        s1 = PostcodeSmoother(h3_resolution=7, k_ring=1)
        s2 = PostcodeSmoother(h3_resolution=7, k_ring=1)

        out1 = s1.fit_transform(
            sample_df, lat_col="latitude", lon_col="longitude"
        )
        s2.fit(sample_df, lat_col="latitude", lon_col="longitude")
        out2 = s2.transform(sample_df, lat_col="latitude", lon_col="longitude")

        pd.testing.assert_series_equal(out1, out2)

    def test_smoothed_values_are_finite(self, sample_df):
        """All smoothed values should be real, finite numbers."""
        smoother = PostcodeSmoother()
        out = smoother.fit_transform(
            sample_df, lat_col="latitude", lon_col="longitude"
        )
        assert out.notna().all(), "smoothed values contain NaN"
        assert np.isfinite(out.values).all(), "smoothed values contain Inf"

    def test_smoothed_values_are_positive(self, sample_df):
        """For positive input values the smoothed result should remain positive."""
        smoother = PostcodeSmoother()
        out = smoother.fit_transform(
            sample_df, lat_col="latitude", lon_col="longitude"
        )
        assert (out > 0).all()

    def test_transform_before_fit_raises(self, sample_df):
        """Calling transform() before fit() must raise RuntimeError."""
        smoother = PostcodeSmoother()
        with pytest.raises(RuntimeError):
            smoother.transform(
                sample_df, lat_col="latitude", lon_col="longitude"
            )

    def test_invalid_resolution_raises(self):
        """Resolutions outside 0–15 should raise ValueError."""
        with pytest.raises(ValueError):
            PostcodeSmoother(h3_resolution=16)
        with pytest.raises(ValueError):
            PostcodeSmoother(h3_resolution=-1)

    def test_invalid_k_ring_raises(self):
        """Negative k_ring values should raise ValueError."""
        with pytest.raises(ValueError):
            PostcodeSmoother(k_ring=-1)

    def test_exposure_weighted_differs_from_unweighted(self, sample_df):
        """Providing exposure weights should produce different results than not."""
        s_unweighted = PostcodeSmoother(h3_resolution=7)
        s_weighted = PostcodeSmoother(h3_resolution=7)

        out_unweighted = s_unweighted.fit_transform(
            sample_df, lat_col="latitude", lon_col="longitude"
        )
        out_weighted = s_weighted.fit_transform(
            sample_df,
            exposure_col="exposure",
            lat_col="latitude",
            lon_col="longitude",
        )

        # They won't be identical when exposures are not uniform.
        assert not out_unweighted.equals(out_weighted)

    def test_get_hex_dataframe(self, sample_df):
        """get_hex_dataframe() should return expected columns after fitting."""
        smoother = PostcodeSmoother()
        smoother.fit(sample_df, lat_col="latitude", lon_col="longitude")
        hex_df = smoother.get_hex_dataframe()

        assert set(hex_df.columns) == {"h3_cell", "mean_value", "total_exposure"}
        assert len(hex_df) > 0

    def test_nan_latlon_falls_back_to_global_mean(self, sample_df):
        """Rows with NaN lat/lon should fall back to the global mean."""
        smoother = PostcodeSmoother()
        smoother.fit(sample_df, lat_col="latitude", lon_col="longitude")

        # A single-row df with NaN coordinates (unresolvable postcode).
        bad_df = pd.DataFrame(
            {
                "postcode": ["ZZ99 9ZZ"],
                "latitude": [float("nan")],
                "longitude": [float("nan")],
            }
        )
        out = smoother.transform(bad_df, lat_col="latitude", lon_col="longitude")

        # The result should equal the global mean (not NaN).
        assert len(out) == 1
        assert np.isfinite(out.iloc[0])
        assert out.iloc[0] == pytest.approx(smoother.global_mean_)

    def test_k_ring_zero_uses_only_target_hex(self, sample_df):
        """k_ring=0 means no neighbours are included – only the target hex."""
        smoother = PostcodeSmoother(h3_resolution=7, k_ring=0)
        out = smoother.fit_transform(
            sample_df, lat_col="latitude", lon_col="longitude"
        )
        assert len(out) == len(sample_df)
        assert out.notna().all()

    def test_inverse_distance_scheme(self, sample_df):
        """inverse_distance weight scheme should produce finite results."""
        smoother = PostcodeSmoother(
            h3_resolution=7, k_ring=2, weight_scheme="inverse_distance"
        )
        out = smoother.fit_transform(
            sample_df, lat_col="latitude", lon_col="longitude"
        )
        assert np.isfinite(out.values).all()


# ─────────────────────────────────────────────────────────────────────────────
# Residual helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestResidualHelpers:
    def test_multiplicative_residual(self):
        obs = pd.Series([1.0, 1.2, 0.8])
        fit = pd.Series([1.0, 1.0, 1.0])
        result = compute_multiplicative_residuals(obs, fit)
        expected = pd.Series([1.0, 1.2, 0.8])
        pd.testing.assert_series_equal(result, expected)

    def test_multiplicative_residual_zero_fitted(self):
        """Division by zero should be handled gracefully via epsilon clip."""
        obs = pd.Series([1.0])
        fit = pd.Series([0.0])
        result = compute_multiplicative_residuals(obs, fit, epsilon=1e-6)
        assert np.isfinite(result.iloc[0])
        assert result.iloc[0] > 0

    def test_additive_residual(self):
        obs = pd.Series([1.0, 1.2, 0.8])
        fit = pd.Series([1.0, 1.0, 1.0])
        result = compute_additive_residuals(obs, fit)
        expected = pd.Series([0.0, 0.2, -0.2])
        pd.testing.assert_series_equal(result, expected, check_exact=False)

    def test_apply_multiplicative(self):
        premium = pd.Series([100.0, 200.0])
        residual = pd.Series([1.1, 0.9])
        result = apply_smoothed_residual(premium, residual, "multiplicative")
        expected = pd.Series([110.0, 180.0])
        pd.testing.assert_series_equal(result, expected, check_exact=False)

    def test_apply_additive(self):
        premium = pd.Series([100.0, 200.0])
        residual = pd.Series([10.0, -20.0])
        result = apply_smoothed_residual(premium, residual, "additive")
        expected = pd.Series([110.0, 180.0])
        pd.testing.assert_series_equal(result, expected, check_exact=False)

    def test_apply_invalid_type_raises(self):
        with pytest.raises(ValueError):
            apply_smoothed_residual(
                pd.Series([1.0]), pd.Series([1.0]), residual_type="invalid"
            )

    def test_summarise_residuals_by_hex(self):
        df = pd.DataFrame(
            {
                "h3_cell": ["abc", "abc", "def"],
                "residual": [1.0, 1.2, 0.9],
            }
        )
        result = summarise_residuals_by_hex(
            df, hex_col="h3_cell", residual_col="residual"
        )
        abc_row = result[result["h3_cell"] == "abc"].iloc[0]
        assert abc_row["mean_residual"] == pytest.approx(1.1)
        assert abc_row["count"] == 2
