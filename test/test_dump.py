# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

'''
Created on Aug 27, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import pytest
from hope._transformer import ASTTransformer
from hope._ast import Module
from hope._dump import Dumper
import hope

def outer_dummy(x):
    return x

class DummyWrapper(object):
    a_a = 0
    a_b = np.array([1])
    def dummy(self, a):
        
        b=np.exp(a)
        c=np.zeros(10, dtype=np.float32)
        c[:] = b + a
        for i in range(3,10):
            b += i
        
        a[1:9] = -b[1:9]
        
        if 1 == 1 and True:
            a[:] = self.a_b[0]
        else:
            a[:] = self.a_a
            outer_dummy(a)
        
        d = True
        while d != True:
            d = False
        e = np.sum(b)
        return hope.exp(e)


@pytest.mark.infrastructure
class TestDumper(object):

    def test_dump_complex(self):
        wrapper = DummyWrapper()
        module = Module(wrapper.dummy.__name__)
        args = [wrapper, np.arange(10)]
        ASTTransformer(module).module_visit(wrapper.dummy, args)
        fkt_str = Dumper().visit(module)
        assert fkt_str is not None
        
        
        
