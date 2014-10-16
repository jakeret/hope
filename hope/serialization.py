# Copyright (c) 2014 ETH Zurich, Institute of Astronomy, Lukas Gamper <lukas.gamper@usystems.ch>

from __future__ import print_function, division, absolute_import, unicode_literals


import os
import pickle

from hope import config


def serialize(obj, name):
    """
    Write a pickled representation of obj to a file named ``name`` inside ``hope.config.prefix``

    :param obj: arbitrary object to serialize
    :type obj: mixed
    :param name: name of the object
    :type name: str
    """

    if not os.path.exists(config.prefix):
        os.makedirs(config.prefix)

    with open(os.path.join(config.prefix, "{0}.pck".format(name)), "wb") as fp:
        pickle.dump(obj, fp)


def unserialize(name):
    """
    Read an object named ``name`` form ``hope.config.prefix``. If the file does not exits ``unserialize`` returns ``None``

    :param name: name of the object
    :type name: str
    :returns: mixed -- unserialized object
    """

    if not os.path.exists(os.path.join(config.prefix, "{0}.pck".format(name))):
        return None

    with open(os.path.join(config.prefix, "{0}.pck".format(name)), "rb") as fp:
        return pickle.load(fp)
