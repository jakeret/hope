# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

'''
Created on Feb 16, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import hope
import pytest
import numpy as np

from test.utilities import setup_module, setup_method, teardown_module
from hope.exceptions import UnsupportedFeatureException

def test_negative_slice_1d_lower():
    def fkt(a): a[-1:] = 1
    ah = np.zeros(10)
    with pytest.raises(UnsupportedFeatureException):
        hope.jit(fkt)(ah)

def test_negative_slice_1d_upper():
    def fkt(a): a[:-1] = 1
    ah = np.zeros(10)
    with pytest.raises(UnsupportedFeatureException):
        hope.jit(fkt)(ah)

def test_negative_slice_1d_both():
    def fkt(a): a[1:-1] = 1
    ah = np.zeros(10)
    with pytest.raises(UnsupportedFeatureException):
        hope.jit(fkt)(ah)

def test_negative_slice_1d_reverse():
    def fkt(a): a[-1:1] = 1
    ah = np.zeros(10)
    with pytest.raises(UnsupportedFeatureException):
        hope.jit(fkt)(ah)

def test_negative_slice_2d_lower():
    def fkt(a): a[-1:, -1:] = 1
    ah = np.zeros((10,10))
    with pytest.raises(UnsupportedFeatureException):
        hope.jit(fkt)(ah)

def test_negative_slice_2d_upper():
    def fkt(a): a[:-1, :-1] = 1
    ah = np.zeros((10,10))
    with pytest.raises(UnsupportedFeatureException):
        hope.jit(fkt)(ah)

def test_negative_slice_2d_both():
    def fkt(a): a[1:-1, 1:-1] = 1
    ah = np.zeros((10,10))
    with pytest.raises(UnsupportedFeatureException):
        hope.jit(fkt)(ah)

def test_negative_slice_2d_reverse():
    def fkt(a): a[-1:1, -1:1] = 1
    ah = np.zeros((10,10))
    with pytest.raises(UnsupportedFeatureException):
        hope.jit(fkt)(ah)

