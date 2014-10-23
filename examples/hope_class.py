# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

'''
A simple example to demonstrate the supported functionality with class methods.

Note: this should be used with precaution as unboxing member variables causes some overhead

Created on Oct 23, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import hope

prefix = "hope_class"

hope.config.keeptemp = True
hope.config.verbose = True
hope.config.prefix = prefix

if os.path.exists(prefix):
    shutil.rmtree(prefix)

@hope.jit
def external(a, b):
    return a + b

class DemoClass(object):
    def __init__(self):
        self.a = 1
        
    @hope.jit
    def run(self, b):
        return self.a + b

    def run_external(self, b):
        return external(self.a, b)


def example():
    dc = DemoClass()
    #jit compile a method of a class
    r = dc.run(2)
    assert r == 3

    #call jitted function from a class
    r = dc.run_external(2)
    assert r == 3
    
    
if __name__ == '__main__':
    example()