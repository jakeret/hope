# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test operators for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope, itertools, pytest, sys, sysconfig, os, shutil

from .utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_augmented_pow(dtype, shape):
    def fkt(a, c):
        c[:] **= a
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(np.uint8, shape), random(dtype, shape)
    ao[ao == 0] += 1
    ah[ah == 0] += 1
    co[co == 0] += 1
    ch[ch == 0] += 1
    co, ch = np.copysign(np.sqrt(np.abs(co)), co).astype(dtype), np.copysign(np.sqrt(np.abs(ch)), ch).astype(dtype)
    ao, ah = np.power(np.abs(ao).astype(np.float64), 1. / co.astype(np.float64)).astype(dtype), np.power(np.abs(ah).astype(np.float64), 1. / ch.astype(np.float64)).astype(dtype)
    fkt(np.abs(ao), co),  hfkt(np.abs(ah), ch)
    assert check(co, ch)

# TODO: fix for np.ulonglong and uint64, std::power produce different results
@pytest.mark.parametrize("dtype,shape", itertools.product([dtype for dtype in dtypes if not dtype in [np.uint64, np.ulonglong]], shapes[1:]))
def test_binary_pow(dtype, shape):
    def fkt(a, c):
        c[:] = a ** 2
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, shape), random(dtype, shape)
    ao, ah = np.copysign(np.sqrt(np.abs(ao)), ao).astype(dtype), np.copysign(np.sqrt(np.abs(ah)), ah).astype(dtype)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)

@pytest.mark.parametrize("dtypea,dtypeb", itertools.product(min_dtypes, min_dtypes))
def test_cross_pow_dtype(dtypea, dtypeb):
    if JENKINS and dtypea == np.float32 and dtypeb == np.int8:
        pytest.skip("Fails on debian: dtypea={0!s}, dtypeb={1!s}".format(dtypea, dtypeb))
    def fkt(a, b, c):
        c[:] = a ** b
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh), (co, ch) = random(dtypea, [100]), random(np.int8, []), random(dtypea, [100])
    if bo == 0 or bh == 0:
        bo, bh = dtypeb(1), dtypeb(1)
    else:
        bo, bh = np.abs(bo).astype(dtypeb), np.abs(bh).astype(dtypeb)
    ao, ah = np.power(np.abs(ao).astype(np.float64), 1. / bo).astype(dtypea), np.power(np.abs(ah).astype(np.float64), 1. / bh).astype(dtypea)
    fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)
    fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)
