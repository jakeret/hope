# HOPE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# HOPE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with HOPE.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function, division, absolute_import, unicode_literals


__all__ = ["jit", "config", "serialize", "unserialize"]
__author__ = "Lukas Gamper, Joel Akeret"
__email__ = "hope@phys.ethz.ch"
__version__ = "0.6.1"
__credits__ = "ETH Zurich, Institute for Astronomy"

from hope import config
from hope.jit import jit
from hope.serialization import serialize, unserialize
from hope.options import enableUnsaveMath, disableUnsaveMath

from hope.exp import exp

# save hope version in compiled files
config.version = __version__
