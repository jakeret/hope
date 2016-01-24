======================================================
HOPE - combine the ease of Python and the speed of C++
======================================================

.. image:: https://badge.fury.io/py/hope.svg
    :target: http://badge.fury.io/py/hope

.. image:: https://travis-ci.org/cosmo-ethz/hope.svg?branch=master
        :target: https://travis-ci.org/cosmo-ethz/hope
        
.. image:: https://coveralls.io/repos/cosmo-ethz/hope/badge.svg?branch=master
        :target: https://coveralls.io/r/cosmo-ethz/hope?branch=master

.. image:: https://img.shields.io/badge/docs-latest-blue.svg?style=flat
        :target: http://hope.readthedocs.org/en/latest

.. image:: http://img.shields.io/badge/arXiv-1410.4345-brightgreen.svg?style=flat
        :target: http://arxiv.org/abs/1410.4345


**HOPE** is a specialized method-at-a-time JIT compiler written in Python for translating Python source code into C++ and compiles this at runtime. In contrast to other existing JIT compliers, which are designed for general purpose, we have focused our development of the subset of the Python language that is most relevant for astrophysical calculations. By concentrating on this subset, **HOPE** is able to achieve the highest possible performance


By using **HOPE**, the user can benefit from being able to write common numerical code in Python and having the performance of compiled implementation. To enable the **HOPE** JIT compilation, the user needs to add a decorator to the function definition. The package does not require additional information, which ensures that **HOPE** is as non-intrusive as possible:

.. code-block:: python

    from hope import jit

    @jit
    def sum(x, y):
        return x + y

        
The **HOPE** package has been developed at ETH Zurich in the `Software Lab of the Cosmology Research Group <http://www.cosmology.ethz.ch/research/software-lab.html>`_ of the `ETH Institute of Astronomy <http://www.astro.ethz.ch/>`_, and is now publicly available at `GitHub <https://github.com/cosmo-ethz/hope>`_. 

Further information on the package can be found in our `paper <http://www.sciencedirect.com/science/article/pii/S2213133714000687>`_,  on `readthedocs.org <http://hope.readthedocs.org/en/latest/>`_  and on our `website <http://hope.phys.ethz.ch>`_. 

Installation
------------

The package has been uploaded to `PyPI <https://pypi.python.org/pypi/hope>`_ and can be installed at the command line via pip::

    $ pip install hope

Or, if you have virtualenvwrapper installed::

    $ mkvirtualenv hope
    $ pip install hope
