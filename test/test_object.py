# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test functions for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import hope
import numpy as np

from test.utilities import check


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


if __name__ == '__main__':
    test_cls_1()
    test_cls_2()
    # test_cls_3()
    test_cls_4()
