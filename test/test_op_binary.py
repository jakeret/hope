# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test operators for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope, itertools, pytest, sys, sysconfig, os, shutil

from .utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module

@pytest.mark.parametrize("dtype,shape", itertools.product([dtype for dtype in dtypes if issubclass(dtype, np.integer) or dtype == int], shapes))
@make_test
def test_binary_or(a, b, c): return a | b

@pytest.mark.parametrize("dtype,shape", itertools.product([dtype for dtype in dtypes if issubclass(dtype, np.integer) or dtype == int], shapes))
@make_test
def test_binary_xor(a, b, c): return a ^ b

@pytest.mark.parametrize("dtype,shape", itertools.product([dtype for dtype in dtypes if issubclass(dtype, np.integer) or dtype == int], shapes))
@make_test
def test_binary_and(a, b, c): return a & b

@pytest.mark.parametrize("dtype,shape", itertools.product([dtype for dtype in dtypes if issubclass(dtype, np.integer) or dtype == int], shapes[1:]))
@make_test
def test_augmented_or(a, b, c): c[:] |= a

@pytest.mark.parametrize("dtype,shape", itertools.product([dtype for dtype in dtypes if issubclass(dtype, np.integer) or dtype == int], shapes[1:]))
@make_test
def test_augmented_xor(a, b, c): c[:] ^= a

@pytest.mark.parametrize("dtype,shape", itertools.product([dtype for dtype in dtypes if issubclass(dtype, np.integer) or dtype == int], shapes[1:]))
@make_test
def test_augmented_and(a, b, c): c[:] &= a
