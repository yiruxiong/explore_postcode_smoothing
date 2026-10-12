"""
core.py – PostcodeSmoother
===========================
The heart of the package.  Uses Uber's H3 hexagonal grid library to smooth
a numeric value (e.g. claim frequency residual) across neighbouring postcodes.

How does H3 smoothing work?
---------------------------
1. Each UK postcode is geocoded to (lat, lon).
2. The lat/lon is converted to an H3 hex cell ID at a chosen *resolution*
   (see H3 resolution table below).
3. Raw values are aggregated to the hex level (e.g. mean claim frequency per
   hex).
4. For every hex, the H3 library gives us its *k-ring* neighbours – a set of
   hexagons within k steps.
5. The smoothed value for a hex is the (optionally distance-weighted) average
   of its own value and those of its neighbours.
6. Each postcode inherits the smoothed value of the hex it sits in.

H3 resolution guide (approximate edge lengths)
-----------------------------------------------
Resolution 5  ~  252 km²  (county level)      – very heavy smoothing
Resolution 6  ~   36 km²  (district level)
Resolution 7  ~    5 km²  (sector level)       ← recommended default
Resolution 8  ~  0.7 km²  (postcode unit level) – light smoothing
Resolution 9  ~  0.1 km²  (street level)       – almost no smoothing

For motor insurance, resolution 7 strikes a good balance between data
stability (enough policies per hex) and geographic granularity.

Weights
-------
By default all neighbours are treated equally (``weight_scheme="uniform"``).
Optionally, ``weight_scheme="inverse_distance"`` weights hexagons that are
closer to the target hex more heavily using the ring distance (0 = self,
1 = immediate neighbours, etc.).

Exposure weighting
------------------
When fitting, each postcode can supply an *exposure* (e.g. earned car years).
The hex-level summary then becomes a weighted mean, so that postcodes with
more data have proportionally more influence.
"""

from __future__ import annotations

import warnings
from typing import Literal

import h3
import numpy as np
import pandas as pd

from .data import resolve_latlon


class PostcodeSmoother:
    """
    Smooth a numeric value across UK postcodes using H3 hexagonal neighbours.

    Parameters
    ----------
    h3_resolution : int
        H3 resolution (0–15).  Resolution 7 (~5 km²) is the recommended
        default for UK motor insurance postcode smoothing.
    k_ring : int
        Number of rings of hexagonal neighbours to include.
        k=1 means the target hex + 6 immediate neighbours (7 total).
        k=2 includes 19 hexagons, k=3 includes 37, etc.
        Larger k = more smoothing = more geographic averaging.
    weight_scheme : "uniform" | "inverse_distance"
        How to weight hexagons when computing the neighbourhood average.
        "uniform"           – every hex in the k-ring contributes equally.
        "inverse_distance"  – hexagons closer to the target are weighted more
                              (weight = 1 / (ring_distance + 1)).
    min_exposure : float
        Hexagons (and their neighbours) with total exposure below this
        threshold are considered *sparse*; their smoothed value is replaced by
        the global mean to avoid noisy estimates.  Set to 0 to disable.

    Attributes
    ----------
    hex_values_ : dict[str, float]
        Fitted hex-level aggregated values (populated by ``fit``).
    global_mean_ : float
        Exposure-weighted global mean (fallback for sparse hexagons).
    """

    def __init__(
        self,
        h3_resolution: int = 7,
        k_ring: int = 1,
        weight_scheme: Literal["uniform", "inverse_distance"] = "uniform",
        min_exposure: float = 0.0,
    ):
        if not (0 <= h3_resolution <= 15):
            raise ValueError("h3_resolution must be between 0 and 15.")
        if k_ring < 0:
            raise ValueError("k_ring must be >= 0.")

        self.h3_resolution = h3_resolution
        self.k_ring = k_ring
        self.weight_scheme = weight_scheme
        self.min_exposure = min_exposure

        # Populated during fit().
        self.hex_values_: dict[str, float] = {}
        self.hex_exposure_: dict[str, float] = {}
        self.global_mean_: float = 0.0
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        postcode_col: str = "postcode",
        value_col: str = "value",
        exposure_col: str | None = None,
        lat_col: str | None = None,
        lon_col: str | None = None,
    ) -> "PostcodeSmoother":
        """
        Compute hex-level aggregated values from the training data.

        Parameters
        ----------
        df : pd.DataFrame
            Training dataset.  Must contain at least ``postcode_col`` and
            ``value_col``.
        postcode_col : str
            Column containing UK postcodes.
        value_col : str
            Column containing the numeric value to smooth (e.g. claim
            frequency, multiplicative GLM residual).
        exposure_col : str | None
            Optional column of exposure weights (e.g. earned car years).
            Enables exposure-weighted hex aggregation.
        lat_col : str | None
            Optional column of pre-computed latitudes.  If both ``lat_col``
            and ``lon_col`` are provided, geocoding via pgeocode is skipped.
            Useful in environments without internet access.
        lon_col : str | None
            Optional column of pre-computed longitudes (see ``lat_col``).

        Returns
        -------
        self : PostcodeSmoother
            Returns the fitted smoother (enables method chaining).
        """
        # ── Step 1: resolve (lat, lon) for every row ────────────────────────
        # Either use supplied columns or geocode via pgeocode.
        lats, lons = resolve_latlon(df, postcode_col, lat_col, lon_col)

        # Attach lat/lon and the target value to a working DataFrame.
        work = df[[postcode_col, value_col]].copy()
        work["_lat"] = lats.values
        work["_lon"] = lons.values

        if exposure_col and exposure_col in df.columns:
            work["_exposure"] = df[exposure_col].values
        else:
            # Default: treat every row as 1 unit of exposure.
            work["_exposure"] = 1.0

        # Drop rows where geocoding failed (e.g. invalid postcode).
        n_before = len(work)
        work = work.dropna(subset=["_lat", "_lon"])
        n_dropped = n_before - len(work)
        if n_dropped > 0:
            warnings.warn(
                f"{n_dropped} row(s) dropped because their postcode could not "
                "be geocoded (lat/lon is NaN).",
                UserWarning,
                stacklevel=2,
            )

        # ── Step 2: assign each row to an H3 hex ────────────────────────────
        # List comprehension is significantly faster than DataFrame.apply for
        # calling h3.latlng_to_cell on large datasets.
        work["_h3_cell"] = [
            h3.latlng_to_cell(lat, lon, self.h3_resolution)
            for lat, lon in zip(work["_lat"], work["_lon"])
        ]

        # ── Step 3: aggregate to hex level (exposure-weighted mean) ─────────
        def weighted_mean(g: pd.DataFrame) -> float:
            """Compute the exposure-weighted mean value within a hex."""
            total_exp = g["_exposure"].sum()
            if total_exp == 0:
                return g[value_col].mean()
            return (g[value_col] * g["_exposure"]).sum() / total_exp

        hex_agg = work.groupby("_h3_cell").apply(weighted_mean).reset_index()
        hex_agg.columns = ["_h3_cell", "_mean_value"]

        hex_exp = work.groupby("_h3_cell")["_exposure"].sum().reset_index()
        hex_exp.columns = ["_h3_cell", "_total_exposure"]

        hex_df = hex_agg.merge(hex_exp, on="_h3_cell")

        # ── Step 4: store as lookup dictionaries ────────────────────────────
        self.hex_values_ = dict(zip(hex_df["_h3_cell"], hex_df["_mean_value"]))
        self.hex_exposure_ = dict(zip(hex_df["_h3_cell"], hex_df["_total_exposure"]))

        # Global fallback: exposure-weighted overall mean.
        total_exp = work["_exposure"].sum()
        if total_exp > 0:
            self.global_mean_ = (
                (work[value_col] * work["_exposure"]).sum() / total_exp
            )
        else:
            self.global_mean_ = work[value_col].mean()

        self._is_fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        postcode_col: str = "postcode",
        lat_col: str | None = None,
        lon_col: str | None = None,
    ) -> pd.Series:
        """
        Return a smoothed value for each row in ``df``.

        For each postcode:
            1. Resolve (lat, lon) – either from supplied columns or pgeocode.
            2. Identify the H3 hex cell.
            3. Collect the k-ring neighbours of that hex.
            4. Average the fitted hex values over the neighbourhood (weighted
               by the chosen weight_scheme and neighbourhood exposures).
            5. Fall back to global mean if exposure is below ``min_exposure``.

        Parameters
        ----------
        df : pd.DataFrame
            Data to score.  Must contain ``postcode_col``.
        postcode_col : str
            Column containing UK postcodes.
        lat_col : str | None
            Optional column of pre-computed latitudes (bypasses pgeocode).
        lon_col : str | None
            Optional column of pre-computed longitudes (bypasses pgeocode).

        Returns
        -------
        pd.Series
            Smoothed values, same length and index as ``df``.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform().")

        # ── Resolve lat/lon ──────────────────────────────────────────────────
        lats, lons = resolve_latlon(df, postcode_col, lat_col, lon_col)

        # ── Assign each row to its H3 cell ───────────────────────────────────
        lat_arr = lats.values
        lon_arr = lons.values

        cells = [
            h3.latlng_to_cell(lat, lon, self.h3_resolution)
            if not (np.isnan(lat) or np.isnan(lon))
            else None
            for lat, lon in zip(lat_arr, lon_arr)
        ]

        # ── Deduplicate: compute the smoothed value once per unique cell ─────
        # Many rows may map to the same H3 cell (e.g. multiple policies in the
        # same postcode district).  Computing the smoothed value once per
        # unique cell and mapping back is much faster than calling
        # _smooth_point() for every row individually.
        unique_cells = set(cells)
        cell_to_smoothed: dict = {}

        for cell in unique_cells:
            if cell is None:
                cell_to_smoothed[None] = self.global_mean_
            else:
                # We need the lat/lon of any representative point in this cell;
                # we use the cell centre provided by H3.
                clat, clon = h3.cell_to_latlng(cell)
                cell_to_smoothed[cell] = self._smooth_point(clat, clon)

        smoothed = [cell_to_smoothed[c] for c in cells]

        return pd.Series(smoothed, index=df.index, name="smoothed_value")

    def fit_transform(
        self,
        df: pd.DataFrame,
        postcode_col: str = "postcode",
        value_col: str = "value",
        exposure_col: str | None = None,
        lat_col: str | None = None,
        lon_col: str | None = None,
    ) -> pd.Series:
        """
        Convenience wrapper: call ``fit`` then ``transform`` on the same data.

        Useful for training-set diagnostics.  For production use, fit on
        training data and transform on hold-out / future data separately.

        Parameters
        ----------
        lat_col, lon_col : str | None
            Optional column names for pre-computed coordinates (see ``fit``).

        Returns
        -------
        pd.Series
            Smoothed values for all rows in ``df``.
        """
        return self.fit(
            df,
            postcode_col=postcode_col,
            value_col=value_col,
            exposure_col=exposure_col,
            lat_col=lat_col,
            lon_col=lon_col,
        ).transform(df, postcode_col=postcode_col, lat_col=lat_col, lon_col=lon_col)

    def get_hex_dataframe(self) -> pd.DataFrame:
        """
        Return a DataFrame with one row per fitted H3 hex cell.

        Useful for mapping: you can pass the hex IDs to a visualisation
        library (e.g. ``pydeck``, ``kepler.gl``) to plot the hex grid.

        Returns
        -------
        pd.DataFrame
            Columns: ``["h3_cell", "mean_value", "total_exposure"]``.
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() before get_hex_dataframe().")

        cells = list(self.hex_values_.keys())
        return pd.DataFrame(
            {
                "h3_cell": cells,
                "mean_value": [self.hex_values_[c] for c in cells],
                "total_exposure": [self.hex_exposure_.get(c, 0.0) for c in cells],
            }
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _smooth_point(self, lat: float, lon: float) -> float:
        """
        Compute the smoothed value for a single geographic point.

        If the point cannot be geocoded (NaN coordinates) or has no neighbours
        in the fitted data, fall back to the global mean.
        """
        # Handle failed geocoding.
        if np.isnan(lat) or np.isnan(lon):
            return self.global_mean_

        # Find the H3 hex for this point.
        target_cell = h3.latlng_to_cell(lat, lon, self.h3_resolution)

        # Collect all hexes within k rings of the target hex.
        # h3.grid_disk(cell, k) returns all cells within k steps including the
        # target cell itself (ring 0).
        neighbourhood = h3.grid_disk(target_cell, self.k_ring)

        # Build lists of values and weights for hexes that have data.
        values = []
        weights = []

        for cell in neighbourhood:
            if cell not in self.hex_values_:
                # No data was observed in this hex during training – skip it.
                continue

            # Ring distance: 0 for target cell, 1 for immediate neighbours, …
            ring_dist = h3.grid_distance(target_cell, cell)

            # Choose weight based on the chosen scheme.
            if self.weight_scheme == "inverse_distance":
                # Weight decreases with distance: 1, 1/2, 1/3, …
                spatial_weight = 1.0 / (ring_dist + 1)
            else:
                # Uniform: all neighbours are equally weighted.
                spatial_weight = 1.0

            # Also weight by the exposure in that hex (more data = more trust).
            exposure_weight = self.hex_exposure_.get(cell, 1.0)

            # Combined weight is the product of spatial and exposure weights.
            combined_weight = spatial_weight * exposure_weight

            values.append(self.hex_values_[cell])
            weights.append(combined_weight)

        # If no data found in the neighbourhood, return the global mean.
        if not values:
            return self.global_mean_

        # Compute the total neighbourhood exposure to decide on the sparse
        # hex fallback.
        total_neighbourhood_exposure = sum(
            self.hex_exposure_.get(c, 0.0)
            for c in neighbourhood
            if c in self.hex_values_
        )

        if (
            self.min_exposure > 0
            and total_neighbourhood_exposure < self.min_exposure
        ):
            # Sparse area: trust the global average more than local noise.
            return self.global_mean_

        # Compute weighted average.
        values_arr = np.array(values)
        weights_arr = np.array(weights)
        return float(np.average(values_arr, weights=weights_arr))
