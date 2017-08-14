# Copyright (C) 2015 ETH Zurich, Institute for Astronomy

'''
Created on Feb 16, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import hope
import pytest
import numpy as np

from .utilities import setup_module, setup_method, teardown_module
from hope.exceptions import UnsupportedFeatureException

def test_negative_idx_1d():
    def fkt(a): a[-1] = 1
    ah = np.zeros(10)
    hope.jit(fkt)(ah)
    assert np.all(ah[:-1] == 0)
    assert ah[-1] == 1


def test_negative_slice_1d_lower():
    def fkt(a): a[-1:] = 1
    ah = np.zeros(10)
    with pytest.raises(UnsupportedFeatureException):
        hope.jit(fkt)(ah)

def test_negative_slice_1d_upper():
    def fkt(a): a[:-1] = 1
    ah = np.zeros(10)
    hope.jit(fkt)(ah)
    assert np.all(ah[:-1] == 1)
    assert ah[-1] == 0

def test_negative_slice_1d_both():
    def fkt(a): a[1:-1] = 1
    ah = np.zeros(10)
    hope.jit(fkt)(ah)
    assert np.all(ah[1:-1] == 1)
    assert ah[-1] == 0
    assert ah[0] == 0
    
def test_negative_idx_2d():
    def fkt(a): a[-1, -1] = 1
    ao, ah = np.zeros((10,10)), np.zeros((10,10))
    fkt(ao)
    hope.jit(fkt)(ah)
    assert np.all(ao == ah)


def test_negative_slice_2d_lower():
    def fkt(a): a[-1:, -1:] = 1
    ah = np.zeros((10,10))
    with pytest.raises(UnsupportedFeatureException):
        hope.jit(fkt)(ah)

def test_negative_slice_2d_upper():
    def fkt(a): a[:-1, :-1] = 1
    ao, ah = np.zeros((10,10)), np.zeros((10,10))
    fkt(ao)
    hope.jit(fkt)(ah)
    assert np.all(ao == ah)

def test_negative_slice_2d_both():
    def fkt(a): a[1:-1, 1:-1] = 1
    ao, ah = np.zeros((10,10)), np.zeros((10,10))
    fkt(ao)
    hope.jit(fkt)(ah)
    assert np.all(ao == ah)

