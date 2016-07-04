#!/usr/bin/env python

import os
import sys
from setuptools.command.test import test as TestCommand
from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = ["-v", "-s"]
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

readme = open("README.rst").read()
history = open("HISTORY.rst").read().replace(".. :changelog:", "")
# during runtime
requires = ["numpy>=1.7", "sympy>=0.7.5"]
# for testing
tests_require = ["pytest>=2.3", "mock>=1.0.1", "tox>=1.8.0",
                 "coverage>=3.7.1", "pytest-cov>=1.8.0"]
# for development and documentation
extras_require = {
    "docs": "Sphinx>=1.2.3",
    "lint": "flake8>=2.2.4",
}

PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
    name="hope",
    version="0.6.1",
    description="A Python Just-In-Time compiler for astrophysical computations",
    long_description=readme + "\n\n" + history,
    author="Lukas Gamper, Joel Akeret",
    author_email="hope@phys.ethz.ch",
    url="http://hope.phys.ethz.ch",
    packages=find_packages(PACKAGE_PATH, "test"),
    package_dir={"hope": "hope"},
    include_package_data=True,
    install_requires=requires,
    tests_require=tests_require,
    extra_requires=extras_require,
    license="GPLv3",
    zip_safe=False,
    keywords="HOPE, JIT compiler, HPC, high performance computing",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Natural Language :: English",
        "Operating System :: MacOS",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Compilers",
    ],
    cmdclass = { "test": PyTest },
)
