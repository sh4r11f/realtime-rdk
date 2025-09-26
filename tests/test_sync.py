import pytest
import sys
from pathlib import Path

# Ensure project root (containing src/) is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ni_sync import int_to_bits


def test_int_to_bits_basic():
    assert int_to_bits(0, 1) == [False]
    assert int_to_bits(1, 1) == [True]
    assert int_to_bits(5, 4) == [True, False, True, False]  # 0101 LSB first


def test_int_to_bits_full_range():
    # For 3 bits max is 7
    assert int_to_bits(7, 3) == [True, True, True]


def test_int_to_bits_errors():
    with pytest.raises(ValueError):
        int_to_bits(-1, 3)
    with pytest.raises(ValueError):
        int_to_bits(8, 3)  # out of range
    with pytest.raises(ValueError):
        int_to_bits(0, 0)
