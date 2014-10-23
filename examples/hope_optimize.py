# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

'''
A simple example to demonstrate the mathematical optimization

http://pythonhosted.org/hope/optimization.html#optimization

Created on Oct 23, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
import hope

prefix = "hope_optimize"

hope.config.keeptemp = True
hope.config.verbose = True
hope.config.optimize = True
hope.config.prefix = prefix

if os.path.exists(prefix):
    shutil.rmtree(prefix)

@hope.jit
def simplify(d):
    r = np.sin(d)**2 + np.cos(d)**2
    return r

@hope.jit
def cse(d):
    r = 2*d + 4*d + 8*d
    return r

@hope.jit
def pow_replacement(d):
    r = d**2 + d**4
    return r

def example():
    # simplified to 1
    r = simplify(0.5)
    assert r == 1

    # reduced to 14 * d
    r = cse(0.5)
    assert r == 7

    # replaces pow and cse:
    # T1 = d * d
    # r = T1 * (1 + T1)
    r = pow_replacement(2)
    assert r == 20


if __name__ == '__main__':
    example()