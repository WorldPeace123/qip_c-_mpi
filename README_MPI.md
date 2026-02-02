# MPI版本量子内积计算程序

## 文件说明

- `test_qip_mpi.cpp` - MPI版本的量子内积计算程序
- `compile_mpi.sh` - 编译脚本
- `vectors_test.txt` - 输入向量文件（两个16维向量）

## MPI安装

如果系统未安装MPI，需要先安装：

### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install mpich
```

### CentOS/RHEL:
```bash
sudo yum install mpich
```

### macOS:
```bash
brew install mpich
```

## 编译

使用编译脚本：
```bash
./compile_mpi.sh
```

或手动编译：
```bash
mpicxx -std=c++11 -O3 -o test_qip_mpi test_qip_mpi.cpp
```

## 运行

### 单进程运行（测试）:
```bash
mpirun -np 1 ./test_qip_mpi
```

### 多进程并行运行:
```bash
mpirun -np 4 ./test_qip_mpi
```

### 使用指定主机:
```bash
mpirun -np 4 -host host1,host2 ./test_qip_mpi
```

## 功能选择

程序启动后可以选择以下计算类型：

1. **向量乘向量** - 计算两个16维向量的量子内积
2. **矩阵乘矩阵** - 计算两个16x16矩阵的量子乘法（并行计算）
3. **向量乘矩阵** - 计算向量与矩阵的量子乘法（并行计算）

## MPI并行策略

### 矩阵乘矩阵 (16x16):
- 总任务数: 256个内积计算
- 任务分配: 按行优先顺序均匀分配给各进程
- 结果收集: 使用MPI_Gatherv收集所有结果到rank 0

### 向量乘矩阵:
- 总任务数: 16个内积计算
- 任务分配: 按列索引均匀分配给各进程
- 结果收集: 使用MPI_Gatherv收集所有结果到rank 0

### 向量乘向量:
- 单个内积计算，所有进程执行相同计算
- 仅rank 0输出结果

## MPI计时

程序使用`MPI_Wtime()`进行高精度计时：
- 计时范围: 从MPI初始化后到MPI_Finalize前
- 输出: 总计算时间（秒）

## 输入文件格式

### vectors_test.txt:
```
2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
```
（前16个数为向量A，后16个数为向量B）

### matrix1.txt / matrix2.txt:
16x16矩阵，每行16个浮点数

### vector_ones.txt:
16维向量

## 输出

结果保存到`result.txt`文件，包含：
- 量子计算结果
- 经典计算结果（用于验证）
- 绝对误差

## 性能优化建议

1. **进程数选择**: 
   - 矩阵乘矩阵: 建议使用4-16个进程
   - 向量乘矩阵: 建议使用2-8个进程
   - 向量乘向量: 单进程即可

2. **编译优化**: 使用`-O3`优化级别

3. **硬件配置**: 确保有足够的CPU核心支持并行计算

## 验证正确性

程序会同时计算量子结果和经典结果，并输出绝对误差。对于正确的实现，误差应该很小（通常<1e-6）。
