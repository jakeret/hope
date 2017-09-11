#!/bin/bash


function convert
{
    jupyter nbconvert --log-level=10 --to html --execute --ExecutePreprocessor.timeout=-1 "$1"
}

pushd benchmarks

rm -f *.so
rm -f src/*.o
rm -rf .hope
rm -rf hope

set -e

convert native_cpp_gen.ipynb
convert "HPC Python.ipynb"
convert fibonacci.ipynb
convert simplify.ipynb
convert star.ipynb
convert julialang.org.ipynb
convert numexpr.ipynb
convert pairwise.ipynb

popd
