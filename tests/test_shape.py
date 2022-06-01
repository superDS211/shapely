from __future__ import annotations

import pytest
import torch
from shapely import ShapeParser, shape

ShapeParser.MAXLEN = 4


data = [
    ((1, 2, 3), "(1, 2, 3)"),
    ([1, 2, 3], "[1, 2, 3]"),
    ([1] * 5, "L(5) [1, 1, 1, 1]"),
    ({1: 1, 2: 2, 3: 3}, "D(3) {1: 1, 2: 2, 3: 3}"),
    ({1: 1, 2: 2, 3: 3, 4: 4, 5: 5}, "D(5) {1: 1, 2: 2, 3: 3, 4: 4}"),
    ([(1, 2), (3, 4)], "[(1, 2), (3, 4)]"),
    (1, "1"),
    (torch.tensor([[1, 2, 3], [1, 2, 3]]), "[2, 3]"),
    (ShapeParser(1), "'shapely.shape.ShapeParser'"),
]


@pytest.mark.parametrize("arg, expected", data)
def test_repr(arg, expected):
    assert repr(shape(arg)) == expected
