# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test functions for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import itertools
import pytest
import copy

from .utilities import random, check, make_test, dtypes, shapes

import hope
from hope.exceptions import UnsupportedFeatureException

# TODO: fix for np.float32
@pytest.mark.parametrize("dtype", [np.float64, float])
def test_func_interp(dtype):
    
    def fkt(x, y, x0, y0, s0, l0, r0):
        y0[:] = np.interp(x0, x, y)
        l0[:] = np.interp(x0 - 1, x, y, left=-1)
        r0[:] = np.interp(x0 + 1, x, y, right=2)
        s0[0] = np.interp(x0[0], x, y)
    hfkt = hope.jit(fkt)
    x0 = np.linspace(0, 1, 50)
    x = np.linspace(0, 1, 10).astype(dtype)
    y = np.linspace(0, 1, 10).astype(dtype)
    xo, xh = np.linspace(0, 1, 50).astype(dtype), np.linspace(0, 1, 50).astype(dtype)
    yo, yh = np.zeros_like(x0), np.zeros_like(x0)
    so, sh = np.zeros_like(x0), np.zeros_like(x0)
    lo, lh = np.zeros_like(x0), np.zeros_like(x0)
    ro, rh = np.zeros_like(x0), np.zeros_like(x0)
    fkt(x, y, xo, yo, so, lo, ro),  hfkt(x, y, xh, yh, sh, lh, rh)
    assert check(xo, xh)
    assert check(yo, yh)
    assert check(so, sh)
    assert check(lo, lh)
    assert check(ro, rh)
    
@pytest.mark.parametrize("dtype", [np.float64, float])
def test_func_interp_bounds(dtype):
    def fkt(x,y,x0,y0):
        y0[:] = np.interp(x0, x, y)
    
    hfkt = hope.jit(fkt)
    x = np.arange(5,15).astype(dtype)
    y = -x
    x0 = np.arange(0, 20).astype(dtype)
    yo, yh = np.zeros_like(x0), np.zeros_like(x0)
    fkt(x, y, x0, yo), hfkt(x, y, x0, yh)
    assert check(yo, yh)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
@make_test
def test_func_sin(a, b, c): return np.sin(a)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
@make_test
def test_func_cos(a, b, c): return np.cos(a)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
@make_test
def test_func_tan(a, b, c): return np.tan(a)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
@make_test
def test_func_sign(a, b, c): return np.sign(a)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
def test_func_arcsin(dtype, shape):
    def fkt(a): 
        return np.arcsin(a)
    
    hfkt = hope.jit(fkt)
    a = 2 * np.random.random(shape) - 1
    b = copy.deepcopy(a)
    co = fkt(a)
    ch = hfkt(b)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
def test_func_arccos(dtype, shape):
    def fkt(a): 
        return np.arccos(a)
    
    hfkt = hope.jit(fkt)
    a = 2 * np.random.random(shape) - 1
    b = copy.deepcopy(a)
    co = fkt(a)
    ch = hfkt(b)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
@make_test
def test_func_arctan(a, b, c): return np.arctan(a)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
def test_func_sinh(dtype, shape):
    def fkt(a): 
        return np.sinh(a)
    
    hfkt = hope.jit(fkt)
    a = np.log(np.abs(random(dtype, shape)[0]) / 2.)
    b = copy.deepcopy(a)
    co = fkt(a)
    ch = hfkt(b)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
def test_func_cosh(dtype, shape):
    def fkt(a): 
        return np.cosh(a)
    
    hfkt = hope.jit(fkt)
    a = np.log(np.abs(random(dtype, shape)[0]) / 2.)
    b = copy.deepcopy(a)
    co = fkt(a)
    ch = hfkt(b)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
@make_test
def test_func_tanh(a, b, c): return np.tanh(a)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
@make_test
def test_func_floor(a, b, c): return np.floor(a)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
@make_test
def test_func_ceil(a, b, c): return np.ceil(a)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
@make_test
def test_func_trunc(a, b, c): return np.trunc(a)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
def test_func_sqrt(dtype, shape):
    def fkt(a): 
        return np.sqrt(a)
    
    hfkt = hope.jit(fkt)
    (ao, ah) = random(dtype, shape)
    ao, ah = np.abs(ao), np.abs(ah)
    co = fkt(ao)
    ch = hfkt(ah)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
@make_test
def test_func_fabs(a, b, c): return np.fabs(a)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
def test_func_exp(dtype, shape):
    def fkt(a): 
        return np.exp(a)
    
    hfkt = hope.jit(fkt)
    ao, ah = np.log(np.abs(random(dtype, shape))).astype(dtype)
    co = fkt(ao)
    ch = hfkt(ah)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product([np.float32, np.float64, float], shapes))
def test_func_log(dtype, shape):
    def fkt(a): 
        return np.log(a)
    
    hfkt = hope.jit(fkt)
    a, b = np.abs(random(dtype, shape))
    co = fkt(a)
    ch = hfkt(b)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes))
def test_func_sum_var(dtype, shape):
    def fkt(a):
        return np.sum(a)
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, shape)
    ao, ah = ao / 1200, ah / 1200
    co, ch = fkt(ao), hfkt(ah)
    assert check(co, ch)
    co, ch = fkt(ao), hfkt(ah)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes))
def test_func_sum_expr(dtype, shape):
    def fkt(a, b):
        return np.sum(a + b)
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh) = random(dtype, shape), random(dtype, shape)
    ao, ah = np.abs(ao) / 2400, np.abs(ah) / 2400
    bo, bh = np.abs(bo) / 2400, np.abs(bh) / 2400
    co, ch = fkt(ao, bo), hfkt(ah, bh)
    assert check(co, ch)
    co, ch = fkt(ao, bo), hfkt(ah, bh)
    assert check(co, ch)

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes))
def test_func_sum_dtype(dtype, shape):
    def fkt(a):
        return np.sum(a, dtype=np.float64)
    
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, shape)
    ao, ah = ao / 1200, ah / 1200
    co, ch = fkt(ao), hfkt(ah)
    assert check(co, ch)
    assert co.dtype == np.float64
    assert ch.dtype == np.float64

def test_func_sum_invalid_args():
    a = np.random.random(1)
    b = np.empty_like(a)
    
    def fkt(a, b):
        return np.sum(a, b)
    
    hfkt = hope.jit(fkt)
    with pytest.raises(UnsupportedFeatureException):
        hfkt(a, b)

    def fkt_axis(a):
        return np.sum(a, axis=0)
    
    hfkt = hope.jit(fkt_axis)
    with pytest.raises(UnsupportedFeatureException):
        hfkt(a)

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes))
def test_func_sum_in_if(dtype, shape):
    def fkt(a):
        if True:
            val = np.sum(a)
        else:
            val = 1
        return val
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, shape)
    ao, ah = ao / 1200, ah / 1200
    co, ch = fkt(ao), hfkt(ah)
    assert check(co, ch)

def test_create_empty_array():
    def fkt(shape):
        return np.empty(shape)
        
    hfkt = hope.jit(fkt)
    assert fkt(5).size == hfkt(5).size

    def fkt_dtype(shape):
        return np.empty(shape, dtype=np.float64)
        
    hfkt = hope.jit(fkt)
    co, ch = fkt(5), hfkt(5)
    assert co.size == ch.size
    assert co.dtype == np.float64
    assert ch.dtype == np.float64

def test_create_zeros_array():
    def fkt(shape):
        return np.zeros(shape)
        
    hfkt = hope.jit(fkt)
    assert fkt(5).size == hfkt(5).size

    def fkt_dtype(shape):
        return np.zeros(shape, dtype=np.float64)
        
    hfkt = hope.jit(fkt)
    co, ch = fkt(5), hfkt(5)
    assert co.size == ch.size
    assert co.dtype == np.float64
    assert ch.dtype == np.float64

def test_create_ones_array():
    def fkt(shape):
        return np.ones(shape)
        
    hfkt = hope.jit(fkt)
    assert fkt(5).size == hfkt(5).size

    def fkt_dtype(shape):
        return np.ones(shape, dtype=np.float64)
        
    hfkt = hope.jit(fkt_dtype)
    co, ch = fkt(5), hfkt(5)
    assert co.size == ch.size
    assert co.dtype == np.float64
    assert ch.dtype == np.float64

def test_create_array_invalid_args():
    def fkt(shape):
        return np.empty(shape, 0)
        
    hfkt = hope.jit(fkt)
    with pytest.raises(UnsupportedFeatureException):
        hfkt(5)

    def fkt_order(shape):
        return np.empty(shape, order="c")
        
    hfkt = hope.jit(fkt_order)
    with pytest.raises(NotImplementedError):
        hfkt(5)

def test_unsupported_np_func():
    def fkt(a):
        return np.alen(a)

    hfkt = hope.jit(fkt)
    a = np.random.random(1)
    
    with pytest.raises(UnsupportedFeatureException):
        hfkt(a)


def test_np_func_invalid_args():
    def fkt(a, b):
        return np.log(a, b)
    
    hfkt = hope.jit(fkt)
    a = np.random.random(1)
    b = np.empty_like(a)
    
    with pytest.raises(UnsupportedFeatureException):
        hfkt(a, b)
    
    def fkt_kwargs(a, b):
        return np.log(a, out=b)
    
    hfkt = hope.jit(fkt_kwargs)
    with pytest.raises(UnsupportedFeatureException):
        hfkt(a, b)
    


# TODO: add tests for remaining functions

