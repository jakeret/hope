# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test operators for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope, itertools, pytest, sys, sysconfig, os, shutil

from .utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_binary_div(dtype, shape):
    def fkt(a, b, c):
        c[:] = a / b
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh), (co, ch) = random(dtype, shape), random(dtype, shape), random(dtype, shape)
    bo[bo == 0] += 1
    bh[bh == 0] += 1
    ro, rh = fkt(ao, bo, co),  hfkt(ah, bh, ch)
    if dtype in [np.float32, np.float64, float]:
        co[co < 1. /  (np.finfo(dtype).max * np.finfo(dtype).resolution)] /= np.finfo(dtype).resolution
        ch[ch < 1. /  (np.finfo(dtype).max * np.finfo(dtype).resolution)] /= np.finfo(dtype).resolution
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_binary_floordiv(dtype, shape):
    def fkt(a, b, c):
        c[:] = a // b
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh), (co, ch) = random(dtype, shape), random(dtype, shape), random(dtype, shape)
    bo[bo == 0] += 1
    bh[bh == 0] += 1
    ro, rh = fkt(ao, bo, co),  hfkt(ah, bh, ch)
    if dtype in [np.float32, np.float64, float]:
        co[co < 1. /  (np.finfo(dtype).max * np.finfo(dtype).resolution)] /= np.finfo(dtype).resolution
        ch[ch < 1. /  (np.finfo(dtype).max * np.finfo(dtype).resolution)] /= np.finfo(dtype).resolution
    assert check(co, ch)

# TODO: fix for np.int8 / np.int8 
@pytest.mark.parametrize("dtypea,dtypeb,dtypec", itertools.product(min_dtypes, min_dtypes, min_dtypes))
def test_cross_div(dtypea, dtypeb, dtypec):
    if dtypea == np.int8 and dtypeb == np.int8:
        pytest.skip("Different behaviour in c++ and python for int8 / int8".format(dtypea, dtypeb))
    def fkt(a, b, c):
        c[:] = a / b
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh), (co, ch) = random(dtypea, [10]), random(dtypeb, [10]), random(dtypec, [10])
    ao, ah, bo, bh = ao.astype(np.float64), ah.astype(np.float64), bo.astype(np.float64), bh.astype(np.float64)
    ao, ah = np.copysign(np.power(np.abs(ao), 1. / 4.), ao).astype(dtypea), np.copysign(np.power(np.abs(ah), 1. / 4.), ah).astype(dtypea)
    bo, bh = np.copysign(np.power(np.abs(bo), 1. / 4.), bo).astype(dtypeb), np.copysign(np.power(np.abs(bh), 1. / 4.), bh).astype(dtypeb)
    bo[bo == 0] += 1
    bh[bh == 0] += 1
    fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)
    fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_augmented_div(dtype, shape):
    def fkt(a, c):
        c[:] /= a
    hfkt = hope.jit(fkt)
    
    (ao, ah) = random(dtype, shape)
    (co, ch) = random(np.float64, shape)
    
    ao[ao == 0] += 1
    ah[ah == 0] += 1
    
    ro, rh = fkt(ao, co),  hfkt(ah, ch)
    if dtype in [np.float32, np.float64, float]:
        co[co < 1. /  (np.finfo(dtype).max * np.finfo(dtype).resolution)] /= np.finfo(dtype).resolution
        ch[ch < 1. /  (np.finfo(dtype).max * np.finfo(dtype).resolution)] /= np.finfo(dtype).resolution
    assert check(co.astype(dtype), ch.astype(dtype))

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_augmented_floordiv(dtype, shape):
    def fkt(a, c):
        c[:] //= a
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, shape), random(dtype, shape)
    ao[ao == 0] += 1
    ah[ah == 0] += 1
    ro, rh = fkt(ao, co),  hfkt(ah, ch)
    if dtype in [np.float32, np.float64, float]:
        co[co < 1. /  (np.finfo(dtype).max * np.finfo(dtype).resolution)] /= np.finfo(dtype).resolution
        ch[ch < 1. /  (np.finfo(dtype).max * np.finfo(dtype).resolution)] /= np.finfo(dtype).resolution
    assert check(co.astype(dtype), ch.astype(dtype))
