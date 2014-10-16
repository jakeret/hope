# Copyright (c) 2013 ETH Zurich, Institute of Astronomy, Lukas Gamper <lukas.gamper@usystems.ch>

from __future__ import print_function, division, absolute_import, unicode_literals


import numpy as np


def exp(x):
    """
    Calculate the exponential of the input.

    :param x: Input data
    :type x: floating point number
    """
    return np.exp(x)
