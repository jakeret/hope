# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test functions for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import hope
import numpy as np

from .utilities import check,setup_module,setup_method,teardown_module


class SubCls(object):
    def __init__(self):
        self.i = 2


class Cls(object):
    def __init__(self):
        self.i = 1
        self.f = 1.
        self.ai = np.array([1, 2, 3], dtype=np.uint32)
        self.af = np.array([1, 2, 3], dtype=np.float32)
        self.res = np.array([0, 0, 0], dtype=np.float32)
        self.obj = SubCls()

    def fkt_cb(self):
        return self.i

    # @hope.jit
    # def fkt_jit(self):
    #   return self.f

    @hope.jit
    def fkt_1(self):
        self.res[:] = self.af

    @hope.jit
    def fkt_2(self):
        return self.fkt_cb()

    # @hope.jit
    # def fkt_3(self):
    #     return self.fkt_jit()

    @hope.jit
    def fkt_4(self):
        return self.obj.i


def test_cls_1():
    inst = Cls()
    inst.fkt_1()
    assert check(inst.af, inst.res)
    inst.af = np.array([1, 2, 3], dtype=np.int_)
    inst.fkt_1()
    assert check(inst.af, inst.res)


def test_cls_2():
    inst = Cls()
    assert check(inst.fkt_2(), inst.i)

# def test_cls_3():
#   inst = Cls()
#   assert check(inst.fkt_3(), inst.f)


def test_cls_4():
    inst = Cls()
    assert check(inst.fkt_4(), inst.obj.i)

def test_member_reference_array():
    class Test(object):
        def __init__(self, n=10):
            self.member = np.ones(n)

        def fkt(self):
            arr = self.member
            return arr
    
        hfkt = hope.jit(fkt)
            
    t1 = Test()
    arr = t1.fkt()
    harr = t1.hfkt()
    assert np.all(arr == harr)
    assert t1.member is harr

def test_member_reference_scalar():
    class Test(object):
        def __init__(self, n=10):
            self.member = n

        def fkt(self):
            a = self.member
            return a
    
        hfkt = hope.jit(fkt)
            
    t1 = Test()
    a = t1.fkt()
    ha = t1.hfkt()
    assert a == ha

def test_member_reference_view():
    class Test(object):
        def __init__(self, n=10):
            self.member = np.ones(n)
            self.out = np.empty(n)

        def fkt(self):
            self.out[:] = self.member
    
        hfkt = hope.jit(fkt)
            
    t1 = Test()
    t1.hfkt()
    assert t1.member is not t1.out
    assert np.all(t1.member == t1.out)

def test_member_reference_member():
    class Test(object):
        def __init__(self, n=10):
            self.member = np.ones(n)
            self.out = np.empty(n)

        def fkt(self):
            self.out = self.member
    
        hfkt = hope.jit(fkt)
            
    t1 = Test()
    t1.fkt()
    assert t1.member is t1.out

def test_member_operation():
    class Test(object):
        def __init__(self):
            self.member = np.ones(3)

        def fkt(self):
            return self.member + 1.0
    
        hfkt = hope.jit(fkt)
            
    t1 = Test()
    a = t1.fkt()
    ah = t1.hfkt()
    assert np.all(a==ah)
