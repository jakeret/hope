# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test functions for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope, itertools, pytest, sys, sysconfig, os, shutil

from .utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module


np_version = tuple(map(int, np.__version__.split(".")))


@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_opt_pow_array(dtype, shape):
    def fkt(a, c):
        c[:] = a ** 2.0 + a ** 4 + a ** 7
    hope.config.optimize = True
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, shape), random(dtype, shape)
    ao, ah = np.copysign(np.power(np.abs(ao), 1. / 8.), ao).astype(dtype), np.copysign(np.power(np.abs(ah), 1. / 8.), ah).astype(dtype)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    hope.config.optimize = False

@pytest.mark.parametrize("dtype", dtypes)
def test_opt_pow_scalar(dtype):
    def fkt(a, c):
        c[0] = a ** 2 + a ** 4 + a ** 7.0
    hope.config.optimize = True
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, []), random(dtype, [1])
    ao, ah = np.copysign(np.power(np.abs(ao), 1. / 8.), ao).astype(dtype), np.copysign(np.power(np.abs(ah), 1. / 8.), ah).astype(dtype)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    hope.config.optimize = False

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_opt_neg_pow_array(dtype, shape):
    def fkt(a, c):
        c[:] = a ** -2 + a ** -4.0 + a ** -7
    hope.config.optimize = True
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, shape), random(dtype, shape)
    ao, ah = np.copysign(np.power(np.abs(ao), 1. / 8.), ao).astype(dtype), np.copysign(np.power(np.abs(ah), 1. / 8.), ah).astype(dtype)
    ao[ao == 0] += 1
    ah[ah == 0] += 1

    if np_version >= (1, 12, 0):
        # numpy 1.12 introduce incompatible change which disallows negative exponents for integers
        ao = ao.astype(np.float)

    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    hope.config.optimize = False

@pytest.mark.parametrize("dtype", dtypes)
def test_opt_neg_pow_scalar(dtype):
    def fkt(a, c):
        c[0] = a ** -2 + a ** -4.0 + a ** -7
    hope.config.optimize = True
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, []), random(dtype, [1])
    ao, ah = np.copysign(np.power(np.abs(ao), 1. / 8.), ao).astype(dtype), np.copysign(np.power(np.abs(ah), 1. / 8.), ah).astype(dtype)
    if ao == 0:
        ao += 1
    if ah == 0:
        ah += 1
    if np_version >= (1, 12, 0):
        # numpy 1.12 introduce incompatible change which disallows negative exponents for integers
        ao = ao.astype(np.float)
    fkt(ao, co), hfkt(ah, ch)
    assert check(co, ch)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    hope.config.optimize = False

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
def test_opt_basic_array(dtype, shape):
    def fkt(a, c):
        c[:] = (a + a) * a - 1 / a
    hope.config.optimize = True
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, shape), random(dtype, shape)
    ao, ah = np.copysign(np.power(np.abs(ao), 1. / 4.), ao).astype(dtype), np.copysign(np.power(np.abs(ah), 1. / 4.), ah).astype(dtype)
    ao[ao == 0] += 1
    ah[ah == 0] += 1
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    hope.config.optimize = False

@pytest.mark.parametrize("dtype", dtypes)
def test_opt_basic_scalar(dtype):
    def fkt(a, c):
        c[0] = (a + a) * a - 1 / a
    hope.config.optimize = True
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, []), random(dtype, [1])
    ao, ah = np.copysign(np.power(np.abs(ao), 1. / 4.), ao).astype(dtype), np.copysign(np.power(np.abs(ah), 1. / 4.), ah).astype(dtype)
    if ao == 0:
        ao += 1
    if ah == 0:
        ah += 1
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    hope.config.optimize = False

@pytest.mark.parametrize("dtype", [float])
def test_opt_basic_scalar(dtype):
    def fkt(a, c):
        c[0] = a**-2
    hope.config.optimize = True
    hfkt = hope.jit(fkt)
    (ao, ah), (co, ch) = random(dtype, []), random(dtype, [1])
    ao, ah = 1. / np.power(np.abs(ao), 1. / 2.).astype(dtype), 1. / np.power(np.abs(ah), 1. / 2.).astype(dtype)
    if ao == 0:
        ao += 1
    if ah == 0:
        ah += 1
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    fkt(ao, co),  hfkt(ah, ch)
    assert check(co, ch)
    hope.config.optimize = False

# TODO: add test for supported functions
# TODO: add test where resulting ast is really tested (pow)

