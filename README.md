# OptiReduce 基准测试指南

本指南介绍如何使用 OptiReduce 在不同网络条件下运行基准测试，通过受控的背景流量来模拟各种网络环境。

## 安装选项

### 选项 1：使用基准测试仓库

我们在基准测试仓库中提供了自动化安装脚本：

```bash
# 克隆基准测试仓库
git clone https://github.com/OptiReduce/benchmark.git
cd benchmark

# 安装基准测试
make install
```



## 背景流量设置

**注意**：`run_background.sh` 脚本可使用 NCCL/Gloo 进行通信。

### 1. 创建网络环境

您可以通过改变背景流量脚本（`run_background.sh`）的工作进程数量来模拟不同的网络环境。这将产生不同的尾延迟与中位数延迟比率，并允许您通过调整 `SIZE` 参数来模拟不同的网络条件：

```bash
Usage: ./run_background.sh -s SIZE -r STARTING_RANK -t TIME [-m MASTER_ADDR] [-p MASTER_PORT] [-d DEVICE]

Options:
  -s SIZE           进程数量
  -r STARTING_RANK  起始排名
  -t TIME           迭代时间（秒）
  -m MASTER_ADDR    NCCL Master 地址（默认：10.0.2.253）
  -p MASTER_PORT    NCCL Master 端口（默认：12355）
  -d DEVICE         网络设备（默认：自动检测）
```

**注意**：如果不指定 `-d DEVICE`，脚本会自动根据 `MASTER_ADDR` 检测网络设备。

#### 示例：低尾延迟环境（p99/p50 = 1.5x）

在任意两个节点上运行以下命令：

```bash
# 在第一个节点上（自动检测网卡）
./run_background.sh -s 4 -r 0 -t 240000 -m 10.0.2.253

# 在第二个节点上（自动检测网卡）
./run_background.sh -s 4 -r 1 -t 240000 -m 10.0.2.253

# 或者手动指定网卡
./run_background.sh -s 4 -r 0 -t 240000 -m 10.0.2.253 -d enp6s27f0np0
```

#### 示例：高尾延迟环境（p99/p50 = 3x）

对于高尾延迟环境，增加 SIZE 参数：

```bash
# 在第一个节点上
./run_background.sh -s 16 -r 0 -t 240000 -m 10.0.2.253

# 在第二个节点上
./run_background.sh -s 16 -r 1 -t 240000 -m 10.0.2.253
```

#### 参数说明

- `-s SIZE`：要生成的进程数量。更高的值会产生更多的背景流量：
    - 4 个进程：创建低尾延迟环境（p99/p50 ≈ 1.5x）
    - 16 个进程：创建高尾延迟环境（p99/p50 ≈ 3x）
- `-r STARTING_RANK`：进程的起始排名（双节点设置中为 0 或 1）
- `-t TIME`：背景流量的持续时间（秒）
- `-m MASTER_ADDR`：NCCL Master 节点 IP 地址（所有节点必须相同）
- `-p MASTER_PORT`：NCCL Master 端口（默认：12355）
- `-d DEVICE`：网络接口名称（可选，如果不指定会自动检测）

**注意：环境配置** 
    大小参数（4 和 16）基于我们的测试环境。您可能需要根据您的环境调整这些值，以实现类似的 p99/p50 延迟比率。监控您的网络条件并相应调整。

## 运行训练



### 使用训练脚本

1. 根据需要启动背景流量（低尾延迟或高尾延迟环境），如上所示

2. 在每个节点上运行训练脚本：

   ### 运行所有通信方案
   ```bash
   RUN_ALL=1 ./run_training.sh <MASTER_ADDR> <RANK> <NODES> <MODEL> [DEV]
   ```
   
   **注意**：`DEV` 参数是可选的。如果不指定，脚本会自动根据 `MASTER_ADDR` 检测网络设备。

   这将按顺序运行以下方案：
   - NCCL with Ring 算法
   - NCCL with Tree 算法

   可用模型：
   - vgg19
   - bert
   - bart
   - roberta
   - gpt2

   双节点设置示例：
   ```bash
   # 运行所有通信方案（自动检测网卡）
   # 在主节点（rank 0）
   RUN_ALL=1 ./run_training.sh 192.168.1.100 0 2 bert

   # 在工作节点（rank 1）
   RUN_ALL=1 ./run_training.sh 192.168.1.100 1 2 bert
   
   # 或者手动指定网卡
   RUN_ALL=1 ./run_training.sh 192.168.1.100 0 2 bert ens17
   ```

   参数：
   - MASTER_ADDR：主节点的 IP 地址
   - RANK：节点排名（0 为主节点，1,2,... 为工作节点）
   - NODES：节点总数
   - MODEL：上面列出的可用模型之一
   - DEV：网络设备名称（可选，如果不指定会自动检测）

## 故障排除

### 自定义训练参数
您可能需要为您的特定用例修改 `run_training.sh` 中的以下参数：
```bash
case $MODEL in
    vgg19)
        BATCH_SIZE=128    # 根据您的 GPU 内存调整
        EPOCHS=150        # 根据模型收敛情况增加/减少
        ;;
    bert)
        BATCH_SIZE=16
        EPOCHS=5
        ;;
    # ... 其他模型
esac
```

### 结果

下表比较了不同通信策略的迭代时间（**s/it**），数值越低越好：

| 模型          | 环境 | NCCL-Ring | NCCL-Tree |
|---------------|-----|-----------|-----------|
| **GPT-2**     | 1.5 | 1.70 s    | 1.52 s    |
|               | 3   | 2.26 s    | 1.91 s    |
| **GPT-2-large** | 1.5 | 7.76 s | 6.46 s |
|               | 3   | 10.12 s   | 9.34 s    |
| **BERT-large** | 1.5 | 5.01 s | 4.24 s |
|               | 3   | 6.53 s    | 5.21 s    |
| **BART-large** | 1.5 | 4.67 s | 4.07 s |
|               | 3   | 6.90 s    | 5.74 s    |
| **RoBERTa-large** | 1.5 | 4.75 s | 4.15 s |
|               | 3   | 7.30 s    | 5.51 s    |
| **Llama-3.2** | 1.5 | 12.92 s | 10.28 s |
|               | 3   | 17.28 s   | 15.72 s   |

### 常见问题

1. **性能下降**
   - 检查 CPU 核心分配
   - 验证线程偏移设置
   - 监控系统是否有其他进程使用分配的核心

2. **训练失败**
   - 确保超时值足够
   - 验证网络设备名称
   - 检查 NCCL Master 地址和端口是否正确配置
   - 确认所有节点可以访问 Master 节点

3. **网络设备问题**
   - 确认正确的设备名称（例如：ens17）

## 支持

对于与部署相关的问题：
1. 详细查看安装日志
2. 在 github 仓库中提交问题

## 许可证

此部署代码是 OptiReduce 项目的一部分。有关许可证信息，请参阅主项目页面。

