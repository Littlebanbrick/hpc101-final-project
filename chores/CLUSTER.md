# 鲲鹏 920F 集群使用指南

深超算（NSCCSZ）灵晟系统，鲲鹏高性能计算全球挑战赛 S2 赛季专用集群。

## 连接方式

### 1. VPN

天融信（TopSec）SSL VPN 客户端，下载 <https://app.topsec.com.cn>。

- 连接地址：<VPN 网关地址已脱敏>
- VPN 账号：`zju_<姓名全拼>`
- 连接后展开「灵晟系统」可见登录节点

### 2. SSH

```bash
# 本机 ~/.ssh/config 已配置别名 kp
ssh kp
# 或直接
ssh <用户名>@<登录入口>
```

- 登录节点入口（IP 已脱敏）是负载均衡，背后有多台登录节点（login01–login10+），
  每台主机指纹不同。`~/.ssh/config` 中已设 `StrictHostKeyChecking no` 绕过。
- 用户名：`<姓名全拼>`
- 认证：SSH 密钥（`~/.ssh/id_ed25519`）或密码

### 3. 作业调度 token

```bash
dlogin        # 输入集群密码获取 token（有时效，过期需重新 dlogin）
dqueue        # 查看队列
djob <JOBID>  # 查看作业状态
```

## 集群环境

| 项 | 值 |
| --- | --- |
| 登录节点 | login01–login10+，麒麟 V10 (Jasmine)，aarch64 |
| CPU | Kunpeng 920F (CPU part `0xd01`)，128 核，4 NUMA × 32 核 |
| 登录节点特性 | NEON (asimd/asimddp)，**无 SVE** |
| 计算节点特性 | armv9 + **SVE**（kplblas 库名含 `armv9p_v2409`） |
| gcc | 10.3.1 (aarch64-linux-gnu) |
| BiSheng 编译器 | 5.1.0 (clang 19.1.7)，路径 `/home/share/eapp_common/chenmiao/BiShengCompiler-5.1.0-aarch64-linux` |
| 作业调度 | 深超算自研 `dsub`/`dqueue`/`dlogin`（非 Slurm） |
| 队列 | `q_kunpeng` |

### 登录节点 NUMA 拓扑

```
NUMA node0: CPU 0-31    node1: CPU 32-63
NUMA node2: CPU 64-95   node3: CPU 96-127
```

### 计算节点拓扑

```
608 核 / 32 NUMA × 38 核 / 2 socket × 304 核 / 2.0 GHz
ARMv9, SVE2, SME
作业分配: 16 vCPU（每 NUMA 1 核, 分布在 16 个不同 NUMA 上）
```

作业请求 cpu=40 或 cpu=128，调度器均只分配 16 vCPU（已验证独占模式也如此）。

## 软件模块

```bash
module use /work_ssd/software/modulefile/modules
module avail                    # 查看所有可用模块
module load kplblas/920FSVE     # 竞赛 BLAS (cblas_dgemm / cblas_domatcopy)
module load armpl/24.10-gcc10.3.1  # ARM Perf Libraries (gcc 版, 登录节点可跑)
```

### kplblas/920FSVE（竞赛指定 BLAS）

- 路径：`/work_ssd/software/soft/tool/kplblas/920FSVE/lib`
- 库名：`libkplblas.so`（链接用 `-lkplblas`，不是 `-lkblas`）
- 符号：`cblas_dgemm`、`cblas_domatcopy`、`cblas_zgemm`、`cblas_zgemm3m`
- **仅计算节点可运行**（armv9 SVE 指令，登录节点会 `Illegal instruction`）
- 头文件 `kblas.h` 不在 kplblas 模块中，从 `/home/share/shenchao_common/kblas/kblas.h` 取

### armpl/24.10-gcc10.3.1（备用，登录节点可编译运行）

- 有 `cblas_dgemm`，但**无 `cblas_domatcopy`**（仅 Fortran 接口 `domatcopy_`）
- 登录节点可运行（NEON），但登录节点性能数据无意义（共享负载干扰）

## 三题编译方式

### CONV（无外部库）

```bash
gcc -O3 -fopenmp bench_conv.c conv2d.c -o conv2d_test -lm
```

### ZGEMM（无外部库）

```bash
gcc -O3 -fopenmp bench_zgemm.c zgemm.c -o zgemm_test -lm
```

### TRSM（需 kplblas）

```bash
module use /work_ssd/software/modulefile/modules
module load kplblas/920FSVE
KPL_LIB=/work_ssd/software/soft/tool/kplblas/920FSVE/lib
KBLAS_H=/home/share/shenchao_common/kblas
export LD_LIBRARY_PATH=$KPL_LIB:$LD_LIBRARY_PATH
gcc -O3 -fopenmp -I"$KBLAS_H" bench_trsm.c trsm.c -o trsm_test \
    -lm -L"$KPL_LIB" -lkplblas -Wl,-rpath,"$KPL_LIB"
```

## 作业提交

### 批处理脚本

```bash
# 脚本头部用 #DSUB 标记调度参数
cat > run.sh <<'EOF'
#!/bin/bash
#DSUB -q q_kunpeng
#DSUB -R cpu=128,mem=256GB
#DSUB -a numa[count=1,distribution=pack]
#DSUB -n my-benchmark
#DSUB -o ~/output_%J.log
... 你的命令 ...
EOF
dsub -s run.sh
```

### 交互式（拿计算节点终端）

```bash
dsub -I -q q_kunpeng -R cpu=128,mem=256GB -a numa[count=1,distribution=pack]
```

### 查看结果

```bash
djob <JOBID>              # 作业状态
cat ~/output_<JOBID>.log  # 输出日志
dpeek <JOBID>             # 实时跟随日志
```

## 重要约束

1. **TRSM 必须在计算节点跑**（登录节点无 SVE → `Illegal instruction`）。
2. **CONV/ZGEMM 登录节点可跑但性能不稳定**（共享节点），正式测速应在计算节点。
3. **计算节点 `numactl -N 1` 导致零并行加速**——深超算自定义内核"亲和域"机制（ccs_agent 管理）强制执行 CPU 分配：线程 affinity 指向分配集之外时全部合流到单核，`sched_setaffinity` 绕不过。赛题官方注明评测资源为**单 NUMA ≤38 核**（该命令在评测机应为正常绑定，零并行为本集群开发容器特有）。run.sh 保留此命令（竞赛规定）。详见 [docs/02-pitfalls.md](docs/02-pitfalls.md#13)。
4. **计算节点脚本必须显式设 PATH**——环境精简，默认 PATH 缺 `/usr/bin`。脚本开头加：
   ```bash
   export PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/batch/cli/bin
   source /usr/share/Modules/init/bash 2>/dev/null
   ```
5. 家目录 `/home/share/<用户名>` 在所有节点共享；`/work_ssd` 仅计算节点可用。
6. `dlogin` token 有时效，过期后所有 `d*` 命令报 "token does not exist"。

## 计算节点资源拓扑

基线结果详见 [bench/baseline_summary.md](bench/baseline_summary.md)。
