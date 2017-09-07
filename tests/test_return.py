# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test functions for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope, itertools, pytest, sys, sysconfig, os, shutil

from .utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module


np_version = tuple(map(int, np.__version__.split(".")))


@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes))
@make_test
def test_return_arr(a, b, c):
    return a


@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes))
def test_return_arr_expr(dtype, shape):
    def fkt(a, c):
        return a ** -2 + a ** -4.0 + a ** -7
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, shape), random(dtype, shape)
    ao, ah = np.copysign(np.power(np.abs(ao), 1. / 8.), ao).astype(dtype), np.copysign(np.power(np.abs(ah), 1. / 8.), ah).astype(dtype)
    if shape:
        ao[ao == 0] += 1
        ah[ah == 0] += 1
    else:
        ao = 1 if ao == 0 else ao
        ah = 1 if ah == 0 else ah
    if np_version >= (1, 12, 0):
        # numpy 1.12 introduce incompatible change which disallows negative exponents for integers
        ao = ao.astype(np.float)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes))
@make_test
def test_return_arrcrt_zeros_dtype(a, b, c):
    d = np.zeros(3, dtype=np.int_)
    d[:] = 2
    d[0] = 1
    return d

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes))
@make_test
def test_return_arrcrt_ones_dtype(a, b, c):
    d = np.ones(3, dtype=np.int_)
    d[:] = 2
    d[0] = 1
    return d

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes))
@make_test
def test_return_arrcrt_zeros(a, b, c):
    d = np.zeros(3)
    d[:] = 2
    d[0] = 1
    return d

# TODO: fix for np.ulonglong, np.longlong and uint64
@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if not dtype in [np.uint64, np.longlong, np.ulonglong]])
def test_return_scalar(dtype):
    def fkt(a): return a
    ao, ah = random(dtype, [])
    ro, rh = fkt(ao), hope.jit(fkt)(ah)
    assert check(ro, rh)

# TODO: fix for np.ulonglong, np.longlong and uint64
@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if not dtype in [np.uint64, np.longlong, np.ulonglong]])
def test_return_arrayscalar(dtype):
    def fkt(a): return a[2]
    ao, ah = random(dtype, [10])
    ro, rh = fkt(ao), hope.jit(fkt)(ah)
    assert check(ro, rh)

# TODO: fix for np.ulonglong, np.longlong and uint64
@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if not dtype in [np.uint64, np.longlong, np.ulonglong]])
def test_return_arrayscalarcpy(dtype):
    def fkt(a): 
        b = a[2]
        return b
    ao, ah = random(dtype, [10])
    ro, rh = fkt(ao), hope.jit(fkt)(ah)
    assert check(ro, rh)

def test_return_const_int():
    def fkt(): return 3
    ro, rh = fkt(), hope.jit(fkt)()
    assert check(ro, rh)

def test_return_const_float():
    def fkt(): return 3.
    ro, rh = fkt(), hope.jit(fkt)()
    assert check(ro, rh)

def test_return_true():
    def fkt():
        return True
    hfkt = hope.jit(fkt)
    ro, rh = fkt(), hfkt()
    assert ro is rh

def test_return_false():
    def fkt():
        return False
    hfkt = hope.jit(fkt)
    ro, rh = fkt(), hfkt()
    assert ro is rh
