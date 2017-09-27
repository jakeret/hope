# encoding: utf-8
from __future__ import print_function, division, absolute_import

# we set a seed, because the legacy hope test code uses random data in somep places which caused
# unforeseeable failures when numpy changed its behaviour handling exponentiation of integer
# types.
import random
import numpy as np

random.seed(12345)
np.random.seed(12345)
