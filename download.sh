#!/bin/bash

# download dataset
wget http://www.zifuture.com:8090/upload/2021/08/mnist.dataset-a2fb0c8f1f91477d871f822419128672.tar.gz --output-document=mnist.dataset.tar.gz
tar -zxvf mnist.dataset.tar.gz
rm -f mnist.dataset.tar.gz

cd openblas_build
./openblas_build_and_install.sh