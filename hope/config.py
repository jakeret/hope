# Copyright (c) 2013 ETH Zurich, Institute of Astronomy, Lukas Gamper <lukas.gamper@usystems.ch>

from __future__ import print_function, division, absolute_import, unicode_literals


from hope.options import get_cxxflags

# Additional compiler flags, formated as array of strings
cxxflags = get_cxxflags()
""" 
List of c++ compiler flags. Normally hope does determing the right flags itself.
"""

#TODO implement
prefix = ".hope"
""" 
Prefix of the folder hope saves all data in.
"""

verbose = False
""" 
Print a intermediate representation of each function during compilation.
"""

optimize = False
"""
Use '''sympy''' to simplify expression and exptract common subexpression detection
"""

keeptemp = False
""" 
Keep the intermediate c++ source and compiler output generated during compilation.
"""

rangecheck = False
""" 
Check if indeces are out of bounds
"""

hopeless = False
""" 
Disable hope. If hope.config.hopeless is True, hope.jit return the original function.
Use this function for debug purpos
"""

# make readable cpp file, but typecasting is not exactly the same as in numpy - this flag is private
_readablecxx = False
