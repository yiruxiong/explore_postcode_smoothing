"""
postcode_smoother
=================
A lightweight UK postcode smoothing package built on Uber's H3 hexagonal
grid (https://www.uber.com/us/en/blog/h3/).

Designed for motor insurance risk premium modelling: instead of treating
each postcode as an isolated point, we borrow signal from nearby postcodes
by averaging their observed values (e.g. claim frequency, loss ratio) across
the H3 hex cells that surround a given location.

Quick-start
-----------
>>> from postcode_smoother import PostcodeSmoother
>>> smoother = PostcodeSmoother(h3_resolution=7)
>>> smoother.fit(df, postcode_col="postcode", value_col="claim_freq")
>>> df["smoothed"] = smoother.transform(df, postcode_col="postcode")
"""

from .core import PostcodeSmoother  # noqa: F401

__all__ = ["PostcodeSmoother"]
__version__ = "0.1.0"
