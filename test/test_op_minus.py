# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test operators for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope, itertools, pytest, sys, sysconfig, os, shutil

from .utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
@make_test
def test_unary_minus(a, b, c): c[:] = -a

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_binary_minus(dtype, shape):
    def fkt(a, b, c):
        c[:] = a - b
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh), (co, ch) = random(dtype, shape), random(dtype, shape), random(dtype, shape)
    ao, ah, bo, bh = (ao / 2.).astype(dtype), (ah / 2.).astype(dtype), (bo / 2.).astype(dtype), (bh / 2.).astype(dtype)
    ro, rh = fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)
    ro, rh = fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_augmented_minus(dtype, shape):
    def fkt(a, c):
        c[:] -= a
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, shape), random(dtype, shape)
    ao, ah, co, ch = (ao / 4.).astype(dtype), (ah / 4.).astype(dtype), (co / 2.).astype(dtype), (ch / 2.).astype(dtype)
    ro, rh = fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    ro, rh = fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
