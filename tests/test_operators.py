# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test operators for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope, itertools, pytest, sys, sysconfig, os, shutil

from .utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module

@pytest.mark.parametrize("dtype,shape", itertools.product([dtype for dtype in dtypes if issubclass(dtype, np.integer) or dtype == int], shapes[1:]))
def test_binary_mod(dtype, shape):
    if JENKINS and dtype == np.int8:
        pytest.skip("Fails on debian: dtype={0!s}".format(dtype))
    def fkt(a, b, c):
        c[:] = a % b
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh), (co, ch) = random(dtype, shape), random(dtype, shape), random(dtype, shape)
    bo[bo == 0] += 1
    bh[bh == 0] += 1
    fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)
    fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product([dtype for dtype in dtypes if issubclass(dtype, np.integer) or dtype == int], shapes[1:]))
def test_binary_lshifts(dtype, shape):
    def fkt(a, b, c):
        c[:] = a << b
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh), (co, ch) = random(dtype, shape), random(dtype, shape), random(dtype, shape)
    bo, bh = (bo % (np.dtype(dtype).itemsize * 8)).astype(dtype), (bh % (np.dtype(dtype).itemsize * 8)).astype(dtype)
    fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)
    fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product([dtype for dtype in dtypes if issubclass(dtype, np.integer) or dtype == int], shapes[1:]))
def test_binary_rshift(dtype, shape):
    def fkt(a, b, c):
        c[:] = a >> b
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh), (co, ch) = random(dtype, shape), random(dtype, shape), random(dtype, shape)
    bo, bh = (bo % (np.dtype(dtype).itemsize * 8)).astype(dtype), (bh % (np.dtype(dtype).itemsize * 8)).astype(dtype)
    fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)
    fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product([dtype for dtype in dtypes if issubclass(dtype, np.integer) or dtype == int], shapes[1:]))
def test_augmented_mod(dtype, shape):
    def fkt(a, c):
        c[:] %= a
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, shape), random(dtype, shape)
    ao[ao == 0] += 1
    ah[ah == 0] += 1
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product([dtype for dtype in dtypes if issubclass(dtype, np.integer) or dtype == int], shapes[1:]))
def test_augmented_lshifts(dtype, shape):
    def fkt(a, c):
        c[:] <<= a
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, [10]), random(dtype, [10])
    ao, ah = (ao % (np.dtype(dtype).itemsize * 8)).astype(dtype), (ah % (np.dtype(dtype).itemsize * 8)).astype(dtype)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product([dtype for dtype in dtypes if issubclass(dtype, np.integer) or dtype == int], shapes[1:]))
def test_augmented_rshift(dtype, shape):
    def fkt(a, c):
        c[:] >>= a
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, [10]), random(dtype, [10])
    ao, ah = (ao % (np.dtype(dtype).itemsize * 8)).astype(dtype), (ah % (np.dtype(dtype).itemsize * 8)).astype(dtype)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
