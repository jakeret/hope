# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test functions for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope, itertools, pytest, sys, sysconfig, os, shutil

from .utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module

def test_opt_pow_array():
    def fkt(a, i):
        return a[i]
    hope.config.rangecheck = True
    hfkt = hope.jit(fkt)
    a = np.array([0, 1, 2])
    try:
        hfkt(a, 0)
        hfkt(a, 1)
        hfkt(a, 2)
    except:
        assert False
    try:
        hfkt(a, -1)
        assert False
    except: 
        pass
    try:
        hfkt(a, 3)
        assert False
    except: 
        pass
    hope.config.rangecheck = False
