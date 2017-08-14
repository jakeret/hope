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
def test_unary_plus(a, b, c): c[:] = +a

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_binary_plus(dtype, shape):
    def fkt(a, b, c):
        c[:] = a + b
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh), (co, ch) = random(dtype, shape), random(dtype, shape), random(dtype, shape)
    ao, ah, bo, bh = (ao / 2.).astype(dtype), (ah / 2.).astype(dtype), (bo / 2.).astype(dtype), (bh / 2.).astype(dtype)
    ro, rh = fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)
    ro, rh = fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)

@pytest.mark.parametrize("dtypea,dtypeb,dtypec", itertools.product(min_dtypes, min_dtypes, min_dtypes))
def test_cross_add_dtype(dtypea, dtypeb, dtypec):
    def fkt(a, b, c):
        c[:] = a + b
    hfkt = hope.jit(fkt)
    dtypeao = np.float64 if dtypea == float else dtypea
    dtypebo = np.float64 if dtypeb == float else dtypeb
    (ao, ah), (bo, bh), (co, ch) = random(dtypeao, [100]), random(dtypebo, [100]), random(dtypec, [100])
    ao, ah = (ao / 2.).astype(dtypea), (ah / 2.).astype(dtypea)
    bo, bh = (bo / 2.).astype(dtypeb), (bh / 2.).astype(dtypeb)
    fkt(ao.astype(dtypea), bo.astype(dtypeb), co),  hfkt(ah.astype(dtypea), bh.astype(dtypeb), ch)
    assert check(co, ch)
    fkt(ao.astype(dtypea), bo.astype(dtypeb), co),  hfkt(ah.astype(dtypea), bh.astype(dtypeb), ch)
    assert check(co, ch)

@pytest.mark.parametrize("dtype", dtypes)
def test_cross_add_scalar(dtype):
    def fkt(a, b, c):
        c[:] = a + b
    hfkt = hope.jit(fkt)
    dtypeo = np.float64 if dtype == float else dtype
    (ao, ah), (bo, bh), (co, ch) = random(dtypeo, [100]), random(dtypeo, []), random(dtypeo, [100])
    ao, ah = (ao / 2.).astype(dtype), (ah / 2.).astype(dtype)
    bo, bh = (bo / 2.).astype(dtype), (bh / 2.).astype(dtype)
    fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)
    fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_cross_add_shape(dtype, shape):
    def fkt(a, b, c):
        c[:] = a + b
    hfkt = hope.jit(fkt)
    dtypeo = np.float64 if dtype == float else dtype
    (ao, ah), (bo, bh), (co, ch) = random(dtypeo, []), random(dtypeo, shape), random(dtypeo, shape)
    ao, ah = (ao / 2.).astype(dtype), (ah / 2.).astype(dtype)
    bo, bh = (bo / 2.).astype(dtype), (bh / 2.).astype(dtype)
    fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)
    fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)
    fkt(bo, ao, co),  hfkt(bh, ah, ch)
    assert check(co, ch)
    fkt(bo, ao, co),  hfkt(bh, ah, ch)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_augmented_plus(dtype, shape):
    def fkt(a, c):
        c[:] += a
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, shape), random(dtype, shape)
    ao, ah, co, ch = (ao / 4.).astype(dtype), (ah / 4.).astype(dtype), (co / 2.).astype(dtype), (ch / 2.).astype(dtype)
    ro, rh = fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    ro, rh = fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
