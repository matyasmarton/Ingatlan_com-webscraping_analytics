"""Feature parsers for the ingatlan.com CSV dump.

Semantics are ported from the legacy cleaning notebook ("Cleaning data 2.ipynb")
with two deliberate corrections:

- Combined rooms parse as ``n + m/2`` (the legacy notebook applied a
  ``Rooms.replace`` table sequentially, so duplicate keys collapsed and the
  first match won, producing wrong values like 3.5 for "1 + 3 fél szoba").
- District labels are the canonical roman-numeral strings ("XIII"), not the
  legacy arabic strings ("13").

All parsers tolerate ``None``/empty/whitespace input without raising.
"""

from __future__ import annotations

import re

import pandas as pd

_ROOMS_COMBINED_RE = re.compile(r"(\d+)\s*\+\s*(\d+)\s*fél\s*szoba")
_ROOMS_PLAIN_RE = re.compile(r"(\d+)\s*szoba")
_PRICE_HUF_RE = re.compile(r"([\d.,]+)\s*M\s*Ft")
_SIZE_SQM_RE = re.compile(r"\d+(?:[.,]\d+)?")
_DISTRICT_RE = re.compile(r"([IVXLCDM]+)\.\s*kerület")


def parse_rooms(s: str) -> float:
    """Parse a rooms string into a float; combined rooms count halves.

    ``" 2 + 1 fél szoba"`` -> 2.5, ``" 12 szoba"`` -> 12.0, else 0.0.
    """
    if s is None:
        return 0.0
    text = str(s).strip()
    if not text:
        return 0.0
    m = _ROOMS_COMBINED_RE.search(text)
    if m:
        return int(m.group(1)) + int(m.group(2)) / 2
    m = _ROOMS_PLAIN_RE.search(text)
    if m:
        return float(m.group(1))
    return 0.0


def parse_price_huf(s: str) -> int | None:
    """Parse ``" 36.5 M Ft "`` into HUF (36_500_000); None when unmatched."""
    if s is None:
        return None
    text = str(s).strip()
    if not text:
        return None
    m = _PRICE_HUF_RE.search(text)
    if not m:
        return None
    return int(round(float(m.group(1).replace(",", ".")) * 1_000_000))


def parse_size_sqm(s: str) -> float | None:
    """Parse ``" 47 m² terület"`` into 47.0; None when no number is present."""
    if s is None:
        return None
    text = str(s).strip()
    if not text:
        return None
    m = _SIZE_SQM_RE.search(text)
    if not m:
        return None
    return float(m.group(0).replace(",", "."))


def parse_district(addr: str) -> str:
    """Extract the canonical roman district label ("XIII"); "other" if absent."""
    if addr is None:
        return "other"
    m = _DISTRICT_RE.search(str(addr))
    if m:
        return m.group(1)
    return "other"


def collapse_rare_labels(s: pd.Series, min_count: int = 2) -> pd.Series:
    """Map labels seen fewer than `min_count` times to "other"."""
    counts = s.value_counts()
    rare = set(counts[counts < min_count].index)
    return s.where(~s.isin(rare), "other")
