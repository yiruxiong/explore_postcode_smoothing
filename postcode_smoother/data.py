"""
data.py – Postcode geocoding helper
====================================
Converts UK postcodes to (latitude, longitude) pairs.

Two approaches are supported:

1. **Automatic (recommended for most users):** relies on the `pgeocode` library,
   which downloads the GeoNames postcode database on first use and caches it
   locally.  Requires a one-time internet connection.

2. **Supply your own mapping:** pass a ``postcode_latlon`` dict or DataFrame to
   ``PostcodeSmoother.fit()`` / ``transform()``.  Useful in restricted
   environments or when you already have a bespoke postcode lookup table
   (e.g. from the OS Open Names dataset or the Royal Mail PAF file).

Why do we need coordinates?
    H3 works in geographic space (lat/lon), so before we can assign a postcode
    to a hexagon we first need to know where it is on the map.
"""

import warnings

import pandas as pd


# We cache the geocoder so it is only initialised once (loading the postcode
# database takes a moment the first time).
_GEOCODER = None


def _get_geocoder():
    """Return (and lazily initialise) the UK geocoder singleton."""
    global _GEOCODER
    if _GEOCODER is None:
        try:
            import pgeocode  # optional dependency
            # 'GB' is the ISO country code for the United Kingdom.
            _GEOCODER = pgeocode.Nominatim("GB")
        except Exception as exc:
            raise ImportError(
                "pgeocode could not be initialised. "
                "Either install it and ensure network access for the first "
                "download, or supply lat/lon columns directly in your "
                "DataFrame (lat_col / lon_col arguments).\n"
                f"Original error: {exc}"
            ) from exc
    return _GEOCODER


def postcodes_to_latlon(postcodes: pd.Series) -> pd.DataFrame:
    """
    Look up latitude and longitude for a Series of UK postcodes via pgeocode.

    Parameters
    ----------
    postcodes : pd.Series
        Raw UK postcodes, e.g. ``["SW1A 2AA", "EC2M 6SQ"]``.
        Mixed formatting (with/without space, upper/lower case) is handled
        automatically.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns ``["postcode", "latitude", "longitude"]``.
        Rows for unrecognised postcodes will have NaN coordinates.

    Notes
    -----
    Requires ``pgeocode`` to be installed and network access for the first
    download of the GeoNames postcode database.  If you cannot use pgeocode,
    supply ``lat_col`` / ``lon_col`` directly to ``PostcodeSmoother.fit()``
    and ``transform()`` instead.
    """
    geocoder = _get_geocoder()

    # Normalise formatting: strip whitespace, convert to upper case.
    clean = postcodes.astype(str).str.strip().str.upper()

    result = geocoder.query_postal_code(clean.tolist())

    return pd.DataFrame(
        {
            "postcode": postcodes.values,
            "latitude": result["latitude"].values,
            "longitude": result["longitude"].values,
        }
    )


def resolve_latlon(
    df: pd.DataFrame,
    postcode_col: str,
    lat_col: str | None,
    lon_col: str | None,
) -> tuple[pd.Series, pd.Series]:
    """
    Return (latitude, longitude) Series for a DataFrame.

    If ``lat_col`` and ``lon_col`` are present in ``df``, they are used
    directly (no geocoding required).  Otherwise, postcodes are geocoded
    via ``postcodes_to_latlon()``.

    Parameters
    ----------
    df : pd.DataFrame
    postcode_col : str
    lat_col : str | None   Column name of pre-computed latitudes, or None.
    lon_col : str | None   Column name of pre-computed longitudes, or None.

    Returns
    -------
    (latitude_series, longitude_series)
    """
    if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
        return df[lat_col], df[lon_col]

    # Fall back to automatic geocoding.
    latlon = postcodes_to_latlon(df[postcode_col])
    return (
        pd.Series(latlon["latitude"].values, index=df.index),
        pd.Series(latlon["longitude"].values, index=df.index),
    )
