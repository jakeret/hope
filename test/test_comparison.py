# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test operators for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope, itertools, pytest, sys, sysconfig, os, shutil

from .utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
@make_test
def test_lt(a, b, c): c[:] = a < b

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
@make_test
def test_gt(a, b, c): c[:] = a > b

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
@make_test
def test_le(a, b, c): c[:] = a <= b

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
@make_test
def test_ge(a, b, c): c[:] = a >= b

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
@make_test
def test_eq(a, b, c): c[:] = a == b

@pytest.mark.parametrize("dtype,shape", itertools.product(dtypes, shapes[1:]))
@make_test
def test_neq(a, b, c): c[:] = a != b

def test_true():
    def fkt():
        if True:
            return 1
        else:
            return 0
    hfkt = hope.jit(fkt)
    ro, rh = fkt(), hfkt()
    assert check(ro, rh)

def test_false():
    def fkt():
        if False:
            return 1
        else:
            return 0
    hfkt = hope.jit(fkt)
    ro, rh = fkt(), hfkt()
    assert check(ro, rh)
