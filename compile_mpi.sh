#!/bin/bash

echo "加载OpenMPI模块..."
module load openmpi

echo "编译MPI版本的量子内积计算程序..."

mpicxx -std=c++11 -O3 -o test_qip_mpi test_qip_mpi.cpp

if [ $? -eq 0 ]; then
    echo "编译成功！"
    echo ""
    echo "运行示例："
    echo "  mpirun -np 4 ./test_qip_mpi"
    echo ""
    echo "使用2个进程运行："
    echo "  mpirun -np 2 ./test_qip_mpi"
else
    echo "编译失败！"
    exit 1
fi
