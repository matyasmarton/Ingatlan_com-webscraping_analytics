"""Unit tests for the data-cleaning parsers."""

import pandas as pd
import pytest

from ingatlan.features import (
    collapse_rare_labels,
    parse_district,
    parse_price_huf,
    parse_rooms,
    parse_size_sqm,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" 2 + 1 fél szoba", 2.5),
        (" 1 + 3 fél szoba", 2.5),
        (" 12 szoba", 12.0),
        (" 2 szoba", 2.0),
        (" 1 + 1 fél szoba", 1.5),
        ("", 0.0),
        ("   ", 0.0),
        (None, 0.0),
        ("garzon", 0.0),
    ],
)
def test_parse_rooms(raw, expected):
    assert parse_rooms(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" 36.5 M Ft ", 36_500_000),
        ("36 M Ft", 36_000_000),
        (" 26.49 M Ft ", 26_490_000),
        ("", None),
        ("   ", None),
        (None, None),
        ("n/a", None),
    ],
)
def test_parse_price_huf(raw, expected):
    assert parse_price_huf(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" 47 m² terület", 47.0),
        (" 66.5 m² terület", 66.5),
        ("", None),
        ("   ", None),
        (None, None),
        ("terület nélkül", None),
    ],
)
def test_parse_size_sqm(raw, expected):
    assert parse_size_sqm(raw) == expected


@pytest.mark.parametrize(
    ("addr", "expected"),
    [
        (" Tahi utca, XIII. kerület", "XIII"),
        ("IV. kerület", "IV"),
        (" Szegedi út 55-57, XIX. kerület", "XIX"),
        ("", "other"),
        ("   ", "other"),
        (None, "other"),
        ("no district here", "other"),
    ],
)
def test_parse_district(addr, expected):
    assert parse_district(addr) == expected


def test_collapse_rare_labels():
    s = pd.Series(["IV", "IV", "V", "XIII", "XIII", "XIII"])
    collapsed = collapse_rare_labels(s, min_count=2)
    assert collapsed.tolist() == ["IV", "IV", "other", "XIII", "XIII", "XIII"]
