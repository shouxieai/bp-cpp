#!/bin/bash

rm -rf OpenBLAS-0.3.17
tar -zxf OpenBLAS-0.3.17.tar.gz
cd OpenBLAS-0.3.17
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../../../lean/OpenBLAS0.3.17 ..
make all -j16 && make install -j16

cd ../../
rm -rf OpenBLAS-0.3.17