# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Tests for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope, itertools, pytest, sys, sysconfig, os, shutil

from test.utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module

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
def test_scalar(dtype):
    def fkt(a, b): a[3] = b
    (ao, ah), (b, _) = random(dtype, [10]), random(dtype, [])
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
def test_slice_1d_1(dtype):
    def fkt(a, b): a[2:5] = b[6:9]
    (ao, ah), (b, _) = random(dtype, [10]), random(dtype, [10])
    fkt(ao, b), hope.jit(fkt)(ah, b)
    assert check(ao, ah)
    fkt(ao, b), hope.jit(fkt)(ah, b)
    assert check(ao, ah)

@pytest.mark.parametrize("dtype", dtypes)
def test_slice_1d_2(dtype):
    def test(a, b): a[:5] = b[3:8]
    (ao, ah), (b, _) = random(dtype, [10]), random(dtype, [10])
    test(ao, b), hope.jit(test)(ah, b)
    assert check(ao, ah)
    test(ao, b), hope.jit(test)(ah, b)
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
