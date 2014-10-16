# Copyright (c) 2013 ETH Zurich, Institute of Astronomy, Lukas Gamper <lukas.gamper@usystems.ch>
from __future__ import print_function, division, absolute_import, unicode_literals


__all__ = ["jit", "config", "serialize", "unserialize"]
__author__ = "Lukas Gamper"
__email__ = "hope@phys.ethz.ch"
__version__ = "0.3.0"
__credits__ = "ETH Zurich, Institute for Astronomy"

from hope import config
from hope.jit import jit
from hope.serialization import serialize, unserialize
from hope.options import enableUnsaveMath, disableUnsaveMath

from hope.exp import exp

# save hope version in compiled files
config.version = __version__
