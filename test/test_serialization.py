# Copyright (C) 2014 ETH Zurich, Institute for Astronomy

'''
Created on Aug 25, 2014

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import pytest
from hope import config
import shutil
from hope import serialization
import os
import tempfile

@pytest.mark.infrastructure
class TestSerialization(object):
    
    def setup(self):
        self.prefix = config.prefix
        config.prefix = tempfile.mkdtemp()
        
    
    def test_serialization(self):
        name = "obj"
        obj = {"test":1}
        serialization.serialize(obj, name)
        assert os.path.join(config.prefix, "obj.pck")
        
        obj2 = serialization.unserialize(name)
        assert obj == obj2
        
    def test_serialize_inexistent_prefix(self):
        name = "obj"
        obj = {"test":1}
        prefix = config.prefix
        config.prefix = os.path.join(prefix, "dir")
        serialization.serialize(obj, name)
        assert os.path.exists(config.prefix)
        assert os.path.join(config.prefix, "obj.pck")
        config.prefix = prefix
    
    def test_unserialize_inexistent(self):
        name = "obj1"
        obj = serialization.unserialize(name)
        assert obj == None
    
    def tear_down(self):
        shutil.rmtree(config.prefix)
        config.prefix = self.prefix
