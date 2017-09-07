# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test control structures for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope
import itertools
import pytest
import sys
import sysconfig
import os
import shutil

from .utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module


def test_ifelse_scalar():
    def fkt(a, b, c):
        if b > 0:
            c[:] = -a
        else:
            c[:] = a
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh), (co, ch) = random(np.int_, [10]), (1, 1), random(np.int_, [10])
    fkt(ao, bo, co), hfkt(ah, bh, ch)
    assert check(co, ch)
    (ao, ah), (bo, bh), (co, ch) = random(np.int_, [10]), (0, 0), random(np.int_, [10])
    fkt(ao, bo, co), hfkt(ah, bh, ch)
    assert check(co, ch)

# @pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, [[10]]))
# @make_test
# def test_for_array(a, b, c):
#     for aa in a:
#         c[0] = a


@pytest.mark.parametrize("dtype", dtypes)
def test_for_range_1(dtype):
    def fkt(a, b, c):
        for i in range(10):
            c[:, i] = a[i, :]
            c[i, 1] = a[0, 1] + b[i, 5]
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh), (co, ch) = random(dtype, [10, 10]), random(
        dtype, [10, 10]), random(dtype, [10, 10])
    ao, ah, bo, bh = (ao / 2.).astype(dtype), (ah / 2.).astype(dtype), (bo /
                                                                        2.).astype(dtype), (bh / 2.).astype(dtype)
    ro, rh = fkt(ao, bo, co), hfkt(ah, bh, ch)
    assert check(co, ch)
    ro, rh = fkt(ao, bo, co), hfkt(ah, bh, ch)
    assert check(co, ch)


@pytest.mark.skipif(sys.version_info.major >= 3, reason="requires python2")
@pytest.mark.parametrize("dtype", dtypes)
def test_for_xrange_1(dtype):
    def fkt(a, b, c):
        for i in xrange(10):
            c[:, i] = a[i, :]
            c[i, 1] = a[0, 1] + b[i, 5]
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh), (co, ch) = random(dtype, [10, 10]), random(
        dtype, [10, 10]), random(dtype, [10, 10])
    ao, ah, bo, bh = (ao / 2.).astype(dtype), (ah / 2.).astype(dtype), (bo /
                                                                        2.).astype(dtype), (bh / 2.).astype(dtype)
    ro, rh = fkt(ao, bo, co), hfkt(ah, bh, ch)
    assert check(co, ch)
    ro, rh = fkt(ao, bo, co), hfkt(ah, bh, ch)
    assert check(co, ch)


@pytest.mark.parametrize("dtype", dtypes)
def test_for_range_2(dtype):
    def fkt(a, b, c):
        for i in range(0, 10):
            c[:, i] = a[i, :]
            c[i, 1] = a[0, 1] + b[i, 5]
    hfkt = hope.jit(fkt)
    (ao, ah), (bo, bh), (co, ch) = random(dtype, [10, 10]), random(
        dtype, [10, 10]), random(dtype, [10, 10])
    ao, ah, bo, bh = (ao / 2.).astype(dtype), (ah / 2.).astype(dtype), (bo /
                                                                        2.).astype(dtype), (bh / 2.).astype(dtype)
    assert check(co, ch)
    ro, rh = fkt(ao, bo, co), hfkt(ah, bh, ch)
    assert check(co, ch)


@pytest.mark.parametrize("dtype", dtypes)
def test_for_iteration_vars(dtype):
    """
    Tests if iteration vars can be used for slicing
    Derived from ufig.plugin.resample
    """
    def fkt(a, b, size_x):
        n = 2
        for dx in range(2 * n + 1):
            b[n + 1:(size_x - n - 1)] += a[dx + 1:(size_x + dx - 2 * n - 1)]

    hfkt = hope.jit(fkt)
    size_x = 10
    ao, ah = random(dtype, [size_x])
    ao = (ao / (2 * size_x + 1)).astype(dtype)
    ah = (ah / (2 * size_x + 1)).astype(dtype)

    bo, bh = np.zeros((size_x), dtype), np.zeros((size_x), dtype)

    fkt(ao, bo, size_x)
    hfkt(ah, bh, size_x)

    assert check(bo, bh)


def test_while():
    def fkt():
        i = 10
        j = 0
        while i > 0:
            j += 1
            i -= 1
        return j
    hfkt = hope.jit(fkt)
    ro, rh = fkt(), hfkt()
    assert check(ro, rh)


def test_if_const():
    def fkt():
        if True:
            a = 1
        else:
            a = 2
        if False:
            b = 7
        else:
            b = 2
        return a + b

    hfkt = hope.jit(fkt)
    ro, rh = fkt(), hfkt()
    assert check(ro, rh)


# TODO: make test for for/wile with array and for/else wihle/else
