# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

"""
Test operators for `hope` module.
"""
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import hope, itertools, pytest, sys, sysconfig, os, shutil

from .utilities import random, check, make_test, JENKINS, min_dtypes, dtypes, shapes, setup_module, setup_method, teardown_module


np_version = tuple(map(int, np.__version__.split(".")))

#hope.config.keeptemp = True

def test_ufig_sin_cos():
    sinTable = np.sin(np.array(range(0, (1 << 11) + 1), dtype=np.float64) * 2. * np.pi / np.float64(1 << 11))
    cosTable = np.cos(np.array(range(0, (1 << 11) + 1), dtype=np.float64) * 2. * np.pi / np.float64(1 << 11))
    rngBuffer = np.random.randint(0, 1<<32, size=(44497,)).astype(np.uint32)

    def fkt_sncn(pt, rng, sinTable, cosTable):
        scL = rng >> (32 - 11)
        scB = np.float64(rng & np.uint32((1 << (32 - 11)) - 1)) / np.float64(1 << (32 - 11))
        scA = 1. - scB
        pt[0] = scA * sinTable[scL] + scB * sinTable[scL + 1]
        pt[1] = scA * cosTable[scL] + scB * cosTable[scL + 1]

    rngBuffer = np.random.randint(0, 1<<32, size=(1000,)).astype(np.uint32)
    pt, hpt = np.array([0., 0.]), np.array([0., 0.])
    hope.config.optimize = True
    hsncn = hope.jit(fkt_sncn)

    for rng in rngBuffer:
        fkt_sncn(pt, rng, sinTable, cosTable)
        hsncn(hpt, rng, sinTable, cosTable)
        assert check(pt, hpt)
        ang = rng / float(1 << 32) * 2. * np.pi
        assert np.all((hpt - np.array([np.sin(ang), np.cos(ang)])) / pt < 1e-5)
    hope.config.optimize = False

def test_ufig_gal_intrinsic():
    try:
        from scipy.special import gammaincinv
    except ImportError:
        pytest.skip("Scipy is not available")

    n = np.uint32(1.88 * np.float64(np.int64(1) << 32) / 10.)
    sersicLTable = np.uint32(n >> np.uint32(32 - 9))
    sersicBTable = np.float32(n - (sersicLTable << np.uint32(32 - 9))) / np.float32(1 << (32 - 9))

    # gamma lookup has 1<<9, 512 elements, and a more precise fitt on 0-1/(1<<3) with 1<<11, 2048 elements
    radiusTable = np.empty(((1 << 9) + 1, (1 << (11 + 1)) + 1, ), dtype=np.float32)
    radiusTable[0][0:(1 << 11)] = (np.power(gammaincinv(2e-15, np.float64(range(0, 1 << 11)) / np.float64(1 << 11)), 1e-15) / 1e-15).astype(np.float32)
    radiusTable[0][(1 << 11):(1 << (11 + 1))] = (np.power(gammaincinv(2e-15, 1. - 1. / np.float64(1 << 3) + np.float64(range(0, 1 << 11)) / np.float64(1 << (11 + 3))), 1e-15) / 1e-15).astype(np.float32)
    radiusTable[0][1 << 11] = (np.power(gammaincinv(2e-15, (1. - 1e-15) / np.float64(1 << 11)), 1e-15) / 1e-15).astype(np.float32)
    radiusTable[0][1 << (11 + 1)] = (np.power(gammaincinv(2e-15, 1. - 1. / np.float64(1 << 3) + (1. - 1e-15) / np.float64(1 << (11 + 3))), 1e-15) / 1e-15).astype(np.float32)

    # TODO: make only one gamma interpolation instead of two
    for i in range(1, (1 << 9) + 1):
        n = 10. * np.float64(i << (32 - 9)) / (np.int64(1) << 32)
        k = gammaincinv(2. * n, 0.5)
        radiusTable[i][0:(1 << 11)] = (np.power(gammaincinv(2. * n, np.float64(range(0, 1 << 11)) / np.float64(1 << 11)), n) / np.power(k, n)).astype(np.float32)
        radiusTable[i][(1 << 11):(1 << (11 + 1))] = (np.power(gammaincinv(2. * n, 1. - 1. / np.float64(1 << 3) + np.float64(range(0, 1 << 11)) / np.float64(1 << (11 + 3))), n) / np.power(k, n)).astype(np.float32)
        radiusTable[i][1 << 11] = (np.power(gammaincinv(2. * n, (1. - 1e-15) / np.float64(1 << 11)), n) / np.power(k, n)).astype(np.float32)
        radiusTable[i][1 << (11 + 1)] = (np.power(gammaincinv(2. * n, 1. - 1. / np.float64(1 << 3) + (1. - 1e-15) / np.float64(1 << (11 + 3))), n) / np.power(k, n)).astype(np.float32)

    def fkt_intrinsic(rng, sersicLTable, sersicBTable, radiusTable):
        drMaski = rng >> (32 - 3) == (1 << 3) - 1
        drKi = rng >> np.uint32(drMaski * 3)
        drLi = (drKi >> (32 - 11)) + np.uint32(drMaski * (1 << 11))
        drBi = np.float64(drKi & ((1 << (32 - 11)) - 1)) / np.float64(1 << (32 - 11))
        drAi = 1. - drBi

        nLi = sersicLTable
        nBi = sersicBTable
        nAi = 1 - nBi

        return drAi * (nAi * radiusTable[nLi, drLi] + nBi * radiusTable[nLi, drLi + 1]) \
           + drBi * (nAi * radiusTable[nLi + 1, drLi] + nBi * radiusTable[nLi + 1, drLi + 1])

    rngBuffer = np.random.randint(0, 1<<32, size=(1000,)).astype(np.uint32)
    hope.config.optimize = True
    hintrinsic = hope.jit(fkt_intrinsic)

    for rng in rngBuffer:
        dr = fkt_intrinsic(rng, sersicLTable, sersicBTable, radiusTable)
        hdr = hintrinsic(rng, sersicLTable, sersicBTable, radiusTable)
        assert check(dr, hdr)
    hope.config.optimize = False


def test_ufig_bincount():

    def fkt_bincount(buffer, x, y, size, size_x, size_y):
        for idx in range(size):
            if x[idx] >= 0 and x[idx] <= size_x and y[idx] >= 0 and y[idx] <= size_y:
                x_bin = np.uint64(x[idx])
                y_bin = np.uint64(y[idx])
                buffer[x_bin, y_bin] += 1.
    hope.config.optimize = True
    hbincount = hope.jit(fkt_bincount)

    x = np.random.uniform(-1, 6, size=(1000,))
    y = np.random.uniform(-1, 6, size=x.shape)
    buffer, hbuffer = np.zeros((5, 5)), np.zeros((5, 5))

    fkt_bincount(buffer, x, y, x.size, 5., 5.)
    hbincount(hbuffer, x, y, x.size, 5., 5.)
    hope.config.optimize = False
    assert check(buffer, hbuffer)

@pytest.mark.parametrize("dtype", [float, np.float32, np.float64])
def test_ufig_star(dtype):
    b = 3.5
    a = 1. / np.sqrt(2. ** (1. / (b - 1.)) - 1.)
    r50 = 2

    center = np.array([10.141, 10.414])
    dims = np.array([20, 20])
    # coefficients generated by http://keisan.casio.com/has10/SpecExec.cgi?id=system/2006/1280624821, 7th order
    x1D = np.array([ \
          0.5 - 0.9491079123427585245262 / 2 \
        , 0.5 - 0.7415311855993944398639 / 2 \
        , 0.5 - 0.4058451513773971669066 / 2 \
        , 0.5 \
        , 0.5 + 0.4058451513773971669066 / 2 \
        , 0.5 + 0.7415311855993944398639 / 2 \
        , 0.5 + 0.9491079123427585245262 / 2 \
    ], dtype=dtype)
    w1D = np.array([ \
          0.1294849661688696932706 / 2 \
        , 0.2797053914892766679015 / 2 \
        , 0.38183005050511894495 / 2 \
        , 0.4179591836734693877551 / 2 \
        , 0.38183005050511894495 / 2 \
        , 0.2797053914892766679015 / 2 \
        , 0.1294849661688696932706 / 2 \
    ], dtype=dtype)
    w2D = np.outer(w1D, w1D)
    def fkt_pdf(density, dims, center, w2D, r50, b, a):
        for x in range(dims[0]):
            for y in range(dims[1]):
                dr = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                density[x, y] = np.sum(w2D * 2 * (b - 1) / (2 * np.pi * (r50 * a)**2) * (1 + (dr / (r50 * a))**2)**(-b))
        return density
    hope.config.optimize = True
    hpdf = hope.jit(fkt_pdf)

    density = np.zeros((dims[0], dims[1]), dtype=dtype)
    fkt_pdf(density, dims, center, w2D, r50, b, a)
    hdensity = np.zeros((dims[0], dims[1]), dtype=dtype)
    hpdf(hdensity, dims, center, w2D, r50, b, a)
    hope.config.optimize = False
#     print("going to sleep")
#     import time
#     time.sleep(10)
    assert check(density, hdensity)

    if np.all(hdensity == 1):
        print("asdf")
    else:
        print("else")
