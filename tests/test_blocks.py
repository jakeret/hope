# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Tests for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import hope
import itertools
import pytest

from .utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module

@pytest.mark.parametrize("dtype", dtypes)
def test_blocks_1(dtype):
    def fkt(a, b, c, d):
        e = b[6:9]
        a[2:5] = e
        a[0] = 0
        a[0] = c[1]
        b[:] = d
    (ao, ah), (bo, bh), (c, _), (d, _) = random(dtype, [10]), random(dtype, [10]), random(dtype, [10]), random(dtype, [10])
    fkt(ao, bo, c, d), hope.jit(fkt)(ah, bh, c, d)
    assert check(ao, ah)
    assert check(bo, bh)
    fkt(ao, bo, c, d), hope.jit(fkt)(ah, bh, c, d)
    assert check(ao, ah)
    assert check(bo, bh)

def test_blocks_2():
    def fkt():
        i = 1
        i += 1
        i = 3
        return i
    hfkt = hope.jit(fkt)
    ro, rh = fkt(), hfkt()
    assert check(ro, rh)

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
@make_test
def test_blocks_localvars(a, b, c):
    x = a
    c[:] = x
