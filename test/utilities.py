# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Tests for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope, itertools, pytest, sys, sysconfig, os, shutil, copy
hope.rangecheck = True

# TODO: differentiate between TravisCI (50min limit) and local + different python versions
# TODO: implement complex
# min_dtypes = [np.int8, np.uint64, np.float32, np.float64]
min_dtypes = [np.int8, np.uint64, np.float64]
# dtypes = [np.longlong, np.ulonglong, np.int8, np.int32, np.uint16, np.uint64, np.float32, np.float64, int, float]

if sys.version_info[0] == 3: #PY3 seems to be slower on TravisCI
    dtypes = [np.ulonglong, np.int8, np.uint64, np.float32, int, float]
else:
    dtypes = [np.ulonglong, np.int8, np.int32, np.uint64, np.float32, int, float]
#shapes = [[], [1200], [30, 13]]
shapes = [[], [3, 4]]

try:
    os.environ["HUDSON_URL"]
    JENKINS = True
except KeyError:
    JENKINS = False

ATOL = {}
ATOL[np.int8] = 1
ATOL[np.int16] = 1
ATOL[np.int32] = 1
ATOL[np.int64] = 1
ATOL[np.uint8] = 1
ATOL[np.uint16] = 1
ATOL[np.uint32] = 1
ATOL[np.uint64] = 1
ATOL[np.float32] = np.finfo(np.float32).resolution
ATOL[np.float64] = np.finfo(np.float64).resolution
ATOL[int] = ATOL[np.int_]
ATOL[float] = ATOL[np.float_]
if not np.bool_ in ATOL:
    ATOL[np.bool_] = 1
ATOL[bool] = ATOL[np.bool_]
if not np.longlong in ATOL:
    ATOL[np.longlong] = 1
if not np.ulonglong in ATOL:
    ATOL[np.ulonglong] = 1


def random(dtype, shape):
    if len(shape) == 0:
        if issubclass(dtype, np.inexact) or dtype == float:
            val = dtype(2 * (np.finfo(dtype).max * np.random.random() - np.finfo(dtype).max / 2))
        else:
            val = np.fromstring(np.random.bytes(np.dtype(dtype).itemsize), dtype=np.dtype(dtype))[0]
            if val == np.iinfo(dtype).min:
                val += 1;
    else:
        if issubclass(dtype, np.inexact) or dtype == float:
            val = 2 * (np.finfo(dtype).max * np.random.random(shape) - np.finfo(dtype).max / 2).astype(dtype)
        else:
            val = np.fromstring(np.random.bytes(np.prod(shape) * np.dtype(dtype).itemsize), dtype=np.dtype(dtype)).reshape(shape)
            val[val == np.iinfo(dtype).min] += 1
    return val, copy.deepcopy(val)


def check(a, b):
    atol = ATOL[a.dtype.type] if hasattr(a, "dtype") else ATOL[type(a)]
    aval = a.astype(np.float64) if hasattr(a, "dtype") else np.float64(a)
    btol = ATOL[b.dtype.type] if hasattr(b, "dtype") else ATOL[type(b)]
    bval = b.astype(np.float64) if hasattr(b, "dtype") else np.float64(b)
    return np.allclose(aval, bval, atol=min(atol, btol))


def make_test(fkt):
    def run(dtype, shape):
        hfkt = hope.jit(fkt)
        (ao, ah), (bo, bh), (co, ch) = random(dtype, shape), random(dtype, shape), random(dtype, shape)
        ro, rh = fkt(ao, bo, co),  hfkt(ah, bh, ch)
        for a, b in [(ao, ah), (bo, bh), (co, ch), (ro, rh)]:
            if not a is None or not b is None:
                assert check(a, b)
        ro, rh = fkt(ao, bo, co),  hfkt(ah, bh, ch)
        for a, b in [(ao, ah), (bo, bh), (co, ch), (ro, rh)]:
            if not a is None or not b is None:
                assert check(a, b)
    return run


def setup_module(module):
    np.seterr(all="raise")


def setup_method(self, method):
    np.random.seed(42)


def teardown_module(module):
    # delete all module in the lib directory to make sure the functions are rebuilt every time
    if os.path.isdir(hope.config.prefix):
        for filename in os.listdir(hope.config.prefix):
            path = os.path.join(hope.config.prefix, filename)
            if os.path.isfile(path) and os.path.splitext(path)[1] in [".pck", ".so", ".pyc", ".py"] and (filename.startswith("fkt_") or filename.startswith("test_")):
                os.remove(path)
