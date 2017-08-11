# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test calls for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import hope
import itertools
import pytest

from .utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module

def fkt_call_local_fun_callback(a, b):
    a[:] = b
@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_call_local_fun(dtype, shape):
    def fkt(a, b, c):
        fkt_call_local_fun_callback(a, b)
        c[:] = a
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh), (co, ch) = random(dtype, shape), random(dtype, shape), random(dtype, shape)
    fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)
    fkt(ao, bo, co),  hfkt(ah, bh, ch)
    assert check(co, ch)

def fkt_call_local_fun_return_callback(a):
    return a
@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_call_local_fun_return(dtype, shape):
    def fkt(a):
        return fkt_call_local_fun_return_callback(a)
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, []), random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert check(co, ch)
    co, ch = fkt(ao), hfkt(ah)
    assert check(co, ch)

@hope.jit
def fkt_call_jit_fun_return_callback(a):
    return a
@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_call_jit_fun_return(dtype, shape):
    def fkt(a):
        return fkt_call_jit_fun_return_callback(a)
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, []), random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert check(co, ch)
    co, ch = fkt(ao), hfkt(ah)
    assert check(co, ch)

def fkt_call_scalar_fun_return_callback(a):
    return a + 1
@pytest.mark.parametrize("dtype", dtypes)
def test_call_scalar_fun_return(dtype):
    def fkt(a):
        return fkt_call_scalar_fun_return_callback(a)
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, []), random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert check(co, ch)
    co, ch = fkt(ao), hfkt(ah)
    assert check(co, ch)

@hope.jit
def fkt_call_scalar_jit_fun_return_callback(a):
    return a + 1
@pytest.mark.parametrize("dtype", dtypes)
def test_call_scalar_jit_fun_return(dtype):
    def fkt(a):
        return fkt_call_scalar_jit_fun_return_callback(a)
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, []), random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert check(co, ch)
    co, ch = fkt(ao), hfkt(ah)
    assert check(co, ch)

@hope.jit
def fkt_recursion_jit_callback(n):
    if n < 2:
        return n
    return fkt_recursion_jit_callback(n - 1) + fkt_recursion_jit_callback(n - 2)
def test_recursion_jit():
    assert fkt_recursion_jit_callback(20) == 6765

@hope.jit
def fkt_call_scalar_jit_multi_callback(a):
    return a + 1
def fkt_call_scalar_jit_multi_fkt(a):
    return fkt_call_scalar_jit_multi_callback(a)
@pytest.mark.parametrize("dtype", dtypes)
def test_fkt_call_scalar_jit_multi(dtype):
    hfkt = hope.jit(fkt_call_scalar_jit_multi_fkt)
    (a, _), (c, _) = random(dtype, []), random(dtype, [])
    c = hfkt(a)
    assert check(c, a + 1)
    c = hfkt(a)
    assert check(c, a + 1)
@pytest.mark.parametrize("dtype", dtypes)
def test_fkt_call_scalar_jit_multi_2(dtype):
    hfkt = hope.jit(fkt_call_scalar_jit_multi_fkt)
    (a, _), (c, _) = random(dtype, []), random(dtype, [])
    c = hfkt(a)
    assert check(c, a + 1)
    c = hfkt(a)
    assert check(c, a + 1)

def fkt_recursion_callback(n):
    if n < 2:
        return n
    return fkt_recursion_callback(n - 1) + fkt_recursion_callback(n - 2)
def test_recursion():
    hope_fkt_recursion_callback = hope.jit(fkt_recursion_callback)
    assert hope_fkt_recursion_callback(20) == 6765

# TODO: make tests for function calls with return types and expr / local vars as arguments
