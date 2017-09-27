# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Tests for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import hope
import itertools, pytest

from .utilities import random, check, make_test, dtypes, shapes, setup_module, setup_method, teardown_module

@pytest.mark.parametrize("dtype", dtypes)
def test_assignment(dtype):
    def fkt(a, b): a[:] = b
    (ao, ah), (b, _) = random(dtype, [10]), random(dtype, [10])
    fkt(ao, b), hope.jit(fkt)(ah, b)
    assert check(ao, ah)
    fkt(ao, b), hope.jit(fkt)(ah, b)
    assert check(ao, ah)

@pytest.mark.parametrize("dtype", dtypes)
def test_index_1d(dtype):
    def fkt(a, b): a[5] = b[3]
    (ao, ah), (b, _) = random(dtype, [10]), random(dtype, [10])
    fkt(ao, b), hope.jit(fkt)(ah, b)
    assert check(ao, ah)
    fkt(ao, b), hope.jit(fkt)(ah, b)
    assert check(ao, ah)

@pytest.mark.parametrize("dtype", dtypes)
def test_index_2d(dtype):
    def fkt(a, b): a[4, 2] = b[3, 4]
    (ao, ah), (b, _) = random(dtype, [5, 5]), random(dtype, [5, 5])
    fkt(ao, b), hope.jit(fkt)(ah, b)
    assert check(ao, ah)
    fkt(ao, b), hope.jit(fkt)(ah, b)
    assert check(ao, ah)

@pytest.mark.parametrize("dtype", dtypes)
def test_index_2d_2(dtype):
    def fkt(a): a[1:4, 1:4] = 1
    (ao, ah) = random(dtype, [5, 5])
    
    fkt(ao), hope.jit(fkt)(ah)
    assert check(ao, ah)
    fkt(ao), hope.jit(fkt)(ah)
    assert check(ao, ah)
    
@pytest.mark.parametrize("dtype", dtypes)
def test_scalar(dtype):
    def fkt(a, b): a[3] = b
    (ao, ah), (b, _) = random(dtype, [10]), random(dtype, [])
    fkt(ao, b), hope.jit(fkt)(ah, b)
    assert check(ao, ah)

@pytest.mark.parametrize("dtype", dtypes)
def test_slice_1d_1(dtype):
    def fkt(a, b): a[2:5] = b[6:9]
    (ao, ah), (b, _) = random(dtype, [10]), random(dtype, [10])
    fkt(ao, b), hope.jit(fkt)(ah, b)
    assert check(ao, ah)
    fkt(ao, b), hope.jit(fkt)(ah, b)
    assert check(ao, ah)

@pytest.mark.parametrize("dtype", dtypes)
def test_slice_1d_2(dtype):
    def fkt(a, b): a[:5] = b[3:8]
    (ao, ah), (b, _) = random(dtype, [10]), random(dtype, [10])
    fkt(ao, b), hope.jit(fkt)(ah, b)
    assert check(ao, ah)
    fkt(ao, b), hope.jit(fkt)(ah, b)
    assert check(ao, ah)

@pytest.mark.parametrize("dtype", dtypes)
def test_slice_1d_3(dtype):
    def fkt(a, b): a[2:] = b[1:9]
    (ao, ah), (b, _) = random(dtype, [10]), random(dtype, [10])
    fkt(ao, b), hope.jit(fkt)(ah, b)
    assert check(ao, ah)
    fkt(ao, b), hope.jit(fkt)(ah, b)
    assert check(ao, ah)

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
@make_test
def test_assign(a, b, c): c[:] = a


@pytest.mark.parametrize("dtype", dtypes)
def test_merged_slice(dtype):
    def fkt(a, b, c):
        for i in range(10):
            c[:, i] = a[i, :]
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh), (co, ch) = random(dtype, [10, 10]), random(dtype, [10, 10]), random(dtype, [10, 10])
    ao, ah, bo, bh = (ao / 2.).astype(dtype), (ah / 2.).astype(dtype), (bo / 2.).astype(dtype), (bh / 2.).astype(dtype)
    fkt(ao, bo, co), hfkt(ah, bh, ch)
    assert check(co, ch)
    fkt(ao, bo, co), hfkt(ah, bh, ch)
    assert check(co, ch)

@pytest.mark.parametrize("dtype", dtypes)
def test_merged_slice_2(dtype):
    def fkt(a, c):
        a[1:] = c[:-1] + c[1:]
        c[:] = a
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, [5]), random(dtype, [5])
    ao, ah = (ao / 4.).astype(dtype), (ah / 4.).astype(dtype)
    co, ch = (co / 4.).astype(dtype), (ch / 4.).astype(dtype)
    fkt(ao, co), hfkt(ah, ch)
    assert check(co, ch)
    fkt(ao, co), hfkt(ah, ch)
    assert check(co, ch)
