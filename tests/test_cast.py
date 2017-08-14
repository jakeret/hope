# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test functions for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope, itertools, pytest, sys, sysconfig, os, shutil

from .utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module

@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if dtype != float])
def test_func_bool_(dtype):
    def fkt(a):
        return np.bool_(a)
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert type(co) == type(ch)

@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if dtype != float])
def test_func_int_(dtype):
    def fkt(a):
        return np.int_(a)
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert type(co) == type(ch)

@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if dtype != float])
def test_func_intc(dtype):
    def fkt(a):
        return np.intc(a)
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert type(co) == type(ch)

@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if dtype != float])
def test_func_int8(dtype):
    def fkt(a):
        return np.int8(a)
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert type(co) == type(ch)

@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if dtype != float])
def test_func_int16(dtype):
    def fkt(a):
        return np.int16(a)
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert type(co) == type(ch)

@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if dtype != float])
def test_func_int32(dtype):
    def fkt(a):
        return np.int32(a)
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert type(co) == type(ch)

@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if dtype != float])
def test_func_int64(dtype):
    def fkt(a):
        return np.int64(a)
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert type(co) == type(ch)

@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if dtype != float])
def test_func_uint8(dtype):
    def fkt(a):
        return np.uint8(a)
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert type(co) == type(ch)

@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if dtype != float])
def test_func_uint16(dtype):
    def fkt(a):
        return np.uint16(a)
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert type(co) == type(ch)

@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if dtype != float])
def test_func_uint32(dtype):
    def fkt(a):
        return np.uint32(a)
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert type(co) == type(ch)

@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if dtype != float])
def test_func_uint64(dtype):
    def fkt(a):
        return np.uint64(a)
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert type(co) == type(ch)

@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if dtype != float])
def test_func_float_(dtype):
    def fkt(a):
        return np.float_(a)
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert type(co) == type(ch)

@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if dtype != float])
def test_func_float32(dtype):
    def fkt(a):
        return np.float32(a)
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert type(co) == type(ch)

@pytest.mark.parametrize("dtype", [dtype for dtype in dtypes if dtype != float])
def test_func_float64(dtype):
    def fkt(a):
        return np.float64(a)
    hfkt = hope.jit(fkt)
    ao, ah = random(dtype, [])
    co, ch = fkt(ao), hfkt(ah)
    assert type(co) == type(ch)
