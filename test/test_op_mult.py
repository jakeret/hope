# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test operators for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope, itertools, pytest, sys, sysconfig, os, shutil

from .utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_binary_mult(dtype, shape):
    def fkt(a, b, c):
        c[:] = a * b
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh), (co, ch) = random(dtype, shape), random(dtype, shape), random(dtype, shape)
    ao = np.copysign(np.sqrt(np.abs(ao.astype(np.float64))).astype(dtype), ao).astype(dtype)
    ah = np.copysign(np.sqrt(np.abs(ah.astype(np.float64))).astype(dtype), ah).astype(dtype)
    bo = np.copysign(np.sqrt(np.abs(bo.astype(np.float64))).astype(dtype), bo).astype(dtype)
    bh = np.copysign(np.sqrt(np.abs(bh.astype(np.float64))).astype(dtype), bh).astype(dtype)
    ro, rh = fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)
    ro, rh = fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_augmented_mult(dtype, shape):
    def fkt(a, c):
        c[:] *= a
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, shape), random(dtype, shape)
    ao, ah = np.copysign(np.power(np.abs(ao), 1. / 4.), ao).astype(dtype), np.copysign(np.power(np.abs(ah), 1. / 4.), ah).astype(dtype)
    co, ch = np.copysign(np.power(np.abs(co), 1. / 4.), co).astype(dtype), np.copysign(np.power(np.abs(ch), 1. / 4.), ch).astype(dtype)
    ro, rh = fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    ro, rh = fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
