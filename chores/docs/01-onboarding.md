# 协作者快速上手

面向新加入的组员。读完本文 + 照做，即可连上鲲鹏 920F 集群并编译运行三道赛题。

## 前置条件

- VPN 账号（天融信 TopSec 客户端，`zju_<姓名全拼>`）
- 集群账号（`<姓名全拼>`，初始密码 <已脱敏>，**首次登录立即改密**）
- 本机 SSH 密钥（`~/.ssh/id_ed25519`），公钥已传到集群 `~/.ssh/authorized_keys`

## 三步连上集群

### 1. 装 VPN

下载 <https://app.topsec.com.cn>，连接地址 <VPN 网关地址已脱敏>，输 VPN 账号密码 + MFA 动态码。连上后展开「灵晟系统」。

### 2. 配 SSH

`~/.ssh/config` 加入：

```
Host kp
  HostName <登录入口 IP 已脱敏>
  User <姓名全拼>
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  LogLevel ERROR
```

登录入口（IP 已脱敏）是负载均衡入口，背后有多台 login 节点，指纹每次不同，故关闭严格校验。

### 3. 连接

```bash
ssh kp
```

## 编译运行三题

### CONV / ZGEMM（无外部库，登录节点可跑）

```bash
cd ~/hpc101/CONV  && gcc -O3 -fopenmp bench_conv.c conv2d.c -o conv2d_test -lm
cd ~/hpc101/ZGEMM && gcc -O3 -fopenmp bench_zgemm.c zgemm.c -o zgemm_test -lm
```

### TRSM（需 kplblas，仅计算节点可跑）

```bash
module use /work_ssd/software/modulefile/modules
module load kplblas/920FSVE
KBLAS_H=/home/share/shenchao_common/kblas
KPL_LIB=/work_ssd/software/soft/tool/kplblas/920FSVE/lib
gcc -O3 -fopenmp -I"$KBLAS_H" bench_trsm.c trsm.c -o trsm_test -lm -L"$KPL_LIB" -lkplblas -Wl,-rpath,"$KPL_LIB"
export LD_LIBRARY_PATH=$KPL_LIB:$LD_LIBRARY_PATH
```

### 运行

竞赛文档规定的运行方式：

```bash
OMP_NUM_THREADS=38 numactl -N 1 ./conv2d_test 4096 6144 39 39 1
```

⚠️ **注意**：当前计算节点作业只分到 16 vCPU（每 NUMA 1 核），`numactl -N 1` 会绑定到单 NUMA 导致零并行加速（38 线程与单线程同速）。官方已注明评测资源为**单 NUMA ≤38 核**，此现象为开发容器配额机制所致，评测机应为正常并行。详见 [02-pitfalls.md](02-pitfalls.md#13)。

## 三条铁律

1. **TRSM 必须在计算节点跑**——登录节点无 SVE，kplblas 会 `Illegal instruction`。
2. **计算节点 PATH 为空**——脚本开头必须显式设 `PATH=/usr/local/bin:/usr/bin:...`。
3. **dlogin token 有时效**——过期后所有 `d*` 命令报 "token does not exist"，重新 `dlogin`。

## 提交计算作业

```bash
dlogin                    # 获取 token
dsub -q q_kunpeng -R cpu=40,mem=128GB -o ~/output_%J.log -s run.sh
djob <JOBID>              # 查状态
cat ~/output_<JOBID>.log  # 看输出
```

计算节点脚本开头必须设 PATH（环境精简）：

```bash
export PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/batch/cli/bin
source /usr/share/Modules/init/bash 2>/dev/null
```

详细命令见 [../CLUSTER.md](../CLUSTER.md)。

## 基线结果

三题基线已在计算节点上跑通（全部 PASS）。详见 [../bench/baseline_summary.md](../bench/baseline_summary.md)。
