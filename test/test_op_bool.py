# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test operators for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope, itertools, pytest, sys, sysconfig, os, shutil

from .utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module

@pytest.mark.parametrize("dtype,shape", [[np.int8, []]])
@make_test
def test_binary_and(a, b, c): return a == 1 and b == 1

@pytest.mark.parametrize("dtype,shape", [[np.int8, []]])
@make_test
def test_binary_or(a, b, c): return a == 1 or b == 1
