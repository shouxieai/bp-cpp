#!/bin/bash

# download OpenBLAS0.3.17
wget http://www.zifuture.com:8090/upload/2021/08/OpenBLAS-0.3.17-01ef2b0812104f3bb6452c1746234e8d.tar.gz --output-document=OpenBLAS-0.3.17.tar.gz
rm -rf OpenBLAS-0.3.17
tar -zxf OpenBLAS-0.3.17.tar.gz
cd OpenBLAS-0.3.17
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=../../../lean/OpenBLAS0.3.17 ..
make all -j16 && make install -j16

cd ../../
rm -rf OpenBLAS-0.3.17
rm -f OpenBLAS-0.3.17.tar.gz