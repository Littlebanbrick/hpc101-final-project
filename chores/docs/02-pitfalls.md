# 踩坑记录

大作业环境搭建过程中遇到的问题，按类别记录。每条格式：**现象 → 根因 → 解法**。

---

## 一、本机环境

### 1. HTTPS 全挂（schannel SEC_E_NO_CREDENTIALS）

**现象**：浏览器 HTTPS 正常，但 `curl`/`git`/`Invoke-WebRequest` 全报 `schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS`。

**根因**：git/curl 默认用 Windows schannel TLS 后端，获取不到凭据；且未读取系统代理（`127.0.0.1:7890`），只有浏览器走了代理。

**解法**：git 切 OpenSSL 后端 + 配代理（写入仓库 local 配置，全局只读改不了）：

```bash
git config --local http.sslBackend openssl
git config --local http.proxy http://127.0.0.1:7890
```

### 2. git push 失败（MSYS signal pipe / Permission denied）

**现象**：`git push` 报 `sh.exe: couldn't create signal pipe, Win32 error 5`，或 `Permission denied (publickey)`。

**根因**：git（MSYS 进程）spawn 子进程时 `ssh` 被解析成 `C:\Program Files\Git\usr\bin\ssh.exe`（MSYS 版），该版本在沙箱内建 signal pipe 失败。而原生 `C:\Windows\System32\OpenSSH\ssh.exe` 不受影响。

**解法**：写 `.git/ssh-wrap.cmd`，用全路径调原生 OpenSSH：

```bat
@echo off
"C:\Windows\System32\OpenSSH\ssh.exe" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 %*
```

设置 `GIT_SSH` 指向该脚本（该文件在 `.git/` 下，不被 git 跟踪）。

---

## 二、集群连接

### 3. VPN 软件换了

**现象**：HPC101 文档写「下载安装 aTrust 客户端」，但 aTrust 连不上。

**根因**：集群更换了 VPN 软件，从 SangFor aTrust 换成**天融信 TopSec SSL VPN**。附件《鲲鹏挑战赛 VPN连接方式.pdf》有新指引，文档未更新。

**解法**：下载 <https://app.topsec.com.cn>，连接 <VPN 网关地址已脱敏>。

### 4. 登录节点指纹反复变化

**现象**：`ssh <登录入口>` 反复报 `REMOTE HOST IDENTIFICATION HAS CHANGED`，每次指纹不同。

**根因**：登录入口（IP 已脱敏）是负载均衡入口，背后有多台 login 节点（login01–login10+），每台主机指纹不同。每次连接被分到不同节点。

**解法**：`~/.ssh/config` 设 `StrictHostKeyChecking no` + `UserKnownHostsFile /dev/null`。

### 5. 「VPN 显示的本机内网地址」不是登录节点

**现象**：VPN 客户端显示一个本机内网地址（已脱敏），但 SSH/端口扫描全不通。

**根因**：该地址是 VPN 给本机分配的内网 IP（ping 自己 TTL=128），不是登录节点。路由表里 `On-link` 标记证实。

**解法**：真正的登录节点在客户端「灵晟系统」分组的登录节点按钮指向（IP 已脱敏）。端口 22 开放。

### 6. SSH 密码用错

**现象**：SSH 登录 `Permission denied`。

**根因**：混用了两套密码——VPN 账号密码（<已脱敏>）≠ 集群高性能账号密码（<已脱敏>）。

**解法**：SSH 用集群高性能账号密码。首次登录后立即 `passwd` 改密。

---

## 三、TRSM 编译（最难的一段）

### 7. KBLAS 找不到

**现象**：`ldconfig -p` 无 kblas，`/opt` 无权限读，`find /` 超时。

**根因**：KBLAS 不在系统标准位置，在共享目录 `/home/share/shenchao_common/kblas/`（含 `kblas.h` + `libkblas.so.25.2.1`）和 `/home/share/eapp_common/blas/`。

### 8. libkblas.so 没有软链

**现象**：`-lkblas` 报 `cannot find -lkblas`。

**根因**：共享目录只有 `libkblas.so.25.2.1`，没有 `libkblas.so` 软链（链接器按 `-l` 找 `lib<name>.so`）。SONAME 是 `libkblas.so.1`，运行时也找不到。

**解法**：在 `~/hpc101/kblaslib/` 建软链 `libkblas.so` → `libkblas.so.25.2.1` 和 `libkblas.so.1` → 同。

### 9. libomp.so 权限被拒

**现象**：链接报 `libomp.so not found`，找到的 `/home/share/fangt/deps/lib/libomp.so` 是 `-rw-------`（属 `admin_perf`），无权读。

**根因**：KBLAS 用 LLVM OpenMP 编译（符号 `__kmpc_*`），需要 `libomp.so`；gcc 的 `-fopenmp` 提供 `libgomp`，符号不兼容。可读的 libomp 在毕昇编译器目录里。

**解法**：用 `/home/share/eapp_common/chenmiao/BiShengCompiler-5.1.0-aarch64-linux/lib/libomp.so`。

### 10. Illegal instruction（OpenMP 运行时冲突）

**现象**：编译成功，运行 `Illegal instruction (core dumped)`。纯 OMP 程序正常，一调 `cblas_dgemm` 就崩。

**根因**：shenchao 那份 KBLAS 是用 SVE 指令编译的（`strings` 含 `dgemm_ncopy_4vl_sve_zip`），登录节点 CPU 无 SVE → 执行 SVE 指令即崩。与 OpenMP 运行时冲突是次要因素。

### 11. kplblas 才是竞赛正确的库

**现象**：shenchao 的 KBLAS 在登录节点跑不了；竞赛 README 写 `-lkblas` 但系统无此库。

**根因**：竞赛期望用的是 `kplblas/920FSVE` 模块（`module load`），库名 `libkplblas.so`（`-lkplblas`），为 armv9 计算节点编译（含 SVE）。只有计算节点能运行。

**解法**：

```bash
module use /work_ssd/software/modulefile/modules
module load kplblas/920FSVE
KPL_LIB=/work_ssd/software/soft/tool/kplblas/920FSVE/lib
KBLAS_H=/home/share/shenchao_common/kblas   # 头文件 kblas.h 不在 kplblas 模块中
gcc -O3 -fopenmp -I"$KBLAS_H" bench_trsm.c trsm.c -o trsm_test -lm -L"$KPL_LIB" -lkplblas -Wl,-rpath,"$KPL_LIB"
```

---

## 四、作业调度

### 12. 作业一直 PENDING

**现象**：`dsub` 提交成功，但状态一直 PENDING，`dnode` 显示节点大量空闲（CPU_FREE=608）。

**根因**：`djob -l` 显示 `vcores need:40 supply:0`——调度器当时有临时问题。重新 `dlogin` 获取 token 后提交成功。鲲鹏专用节点为 `cn23033-23036`（共 4 台），通过 `dadmin label show` 可查 `queue_q_kunpeng` 标签。

### 13. 计算节点 numactl -N 导致 14× 性能下降

**现象**：基线代码在计算节点上用 `OMP_NUM_THREADS=38 numactl -N 1`（竞赛文档规定命令）只有 1.04 GFLOPS，而去掉 `numactl` 后 14.8 GFLOPS——差 14 倍。

**关键实验**（OMP=1 对照）：

| 配置 | OMP | 时间(ms) | GFLOPS | 加速比 |
| --- | --- | --- | --- | --- |
| 无 numactl | 1 | 72180 | 1.04 | 1× |
| 无 numactl | 38 | 5205 | 14.5 | 14× ✓ |
| numactl -N 1 | 1 | 71927 | 1.05 | 1× |
| numactl -N 1 | 38 | 72296 | 1.04 | **0×** ✗ |

`numactl -N 1` 时 38 线程与单线程同速——**零并行加速**。

**根因**（2026-08-29 实测确认，深超算自定义内核的 CPU 分配机制）：
- `-N` 是 `--cpunodebind`（绑 CPU），不是 `--membind`（绑内存）。`numactl -N 1` 把进程 affinity 设为 NUMA 1 的 38 核（38-75），`/proc/PID/status` 也如此显示——但 38 个线程实际全部合流到单个核（不同轮次观测到 CPU 38/39）。代码内 `sched_setaffinity` 设回分配核返回 0 但无效。
- cgroup 为纯 v1（无 v2/cpu.max）。根层级可读：`cfs_quota_us=-1`、`cpuset.cpus=0-607`——标准机制均无限制。
- 但存在非标准内核文件：`cpu.affinity_domain_mask`(0x7)、`cpu.affinity_period_ms`(2000)、`cpu.affinity_stat`、`cpu.dynamic_affinity_mode`(0)——深超算自定义"亲和域"扩展；`affinity_stat` 显示作业的 16 个 vCPU 被组织成 4/8/16 核三层调度域。
- 作业容器 cgroup 在 `/sys/fs/cgroup/cpu,cpuacct/batch/` 下，目录 0700 属主 `ccs_agent`，容器内不可读——限制的确切参数无法获取。
- 结论：CPU 分配由 ccs_agent/自定义内核层强制执行，用户态（numactl、sched_setaffinity）绕不过；线程 affinity 一旦指向分配集之外，全部合流到单核。零并行只在计算作业容器内出现；登录节点直跑 `numactl -N 1` 正常（本队 login08 与友队 login06 均约 35 GFLOPS）。
- 补充实验（2026-08-30）：`dsub -R cpu=16 -a numa[count=1,distribution=pack]` 可拿到单 NUMA 连续核（Cpus_allowed_list=0-15），但 `affinity_stat` 的 zone hot 仍为旧散核集——亲和域未随 pack 分配更新，16 线程仍合流单核（OMP=16 = 1.04 GFLOPS）。**本集群上任何 pack/单 NUMA 分配都无法获得多核吞吐；唯一可用并行形态是默认散核分配**。
- 评测环境（官方注明单 NUMA ≤38 核）无此问题——此为深超算开发容器特有现象。

**注意**：竞赛文档明确规定 `OMP_NUM_THREADS=38 numactl -N 1`，**run.sh 必须保留此命令**。

**本地调试时**用默认散核分配（不加 `-a numa`），去掉 numactl 以获得真实并行数据：
```bash
OMP_NUM_THREADS=38 ./xxx  # 调试用, run.sh 提交版保留 numactl
```

### 14. 计算节点 PATH 为空

**现象**：作业脚本在计算节点上报 `hostname: command not found`、`gcc: command not found`、`numactl: command not found`。

**根因**：计算节点环境精简，PATH 只有 `/usr/share/Modules/bin` 和 `/opt/batch/cli/bin`，没有 `/usr/bin` 等。

**解法**：脚本开头显式设置：

```bash
export PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/batch/cli/bin
source /usr/share/Modules/init/bash 2>/dev/null
module use /work_ssd/software/modulefile/modules
```

### 15. 共租户争用导致测量大幅失真（「节点天气」）

**现象**：同一个二进制（库 cblas_dtrsm 对照组）不同 job 之间 GFLOPS 差 4-10 倍：21:30 的 job 里 232/434/214 GFLOPS，22:03 起的连续多个 job 里只剩 21/86/45。自己的实现也随堵车降 1.5-2×（幅度小得多）。

**根因**：608 核节点上 vCPU 是独占的（cgroup 保证），但**内存带宽/内存控制器是全节点共享**——共租户大流量作业会把访存受限的代码拖慢。kblas 多线程内部用 libomp 自旋 barrier，对这种争用特别敏感（崩 4-10×）；libgomp + 单线程 BLAS 调用的结构无内部 barrier，基本免疫（实测降 ~1.5×）。

**解法**：测量前先跑「天气探针」，跨 job 的数据不直接比较：
```bash
# quiet 基线: ~156 GFLOPS (OMP=16)。低于 120 = 节点堵车，测量无意义
OMP_NUM_THREADS=16 ./exp_dgemm 2048 2048 2048 2
```
关键对比实验放在**同一个 job 内交错运行**（v5 vs 库各跑 3 轮取最大值），或挂看门狗 job 每 150s 探测、检测到 quiet 窗口再自动跑对比（`~/hpc101/trsm_watchdog.sh`）。

### 16. 亲和域集群的 numactl 合流「标记」机制（2026-09-03，TRSM v7-v10 血泪）

**现象**：评测命令 `OMP_NUM_THREADS=38 numactl -N 1` 下，38 线程并行比单线程串行慢 5-30×（v6 实测 40s，v7 惨到 130s，而串行只要 7.6s）。

**机制**（黑盒推断，多组实验交叉验证）：
- 重度超订（大量线程在掩码内持续运行百毫秒级，如 38 线程 × ~180ms）触发**粘性标记**：之后本作业/容器的多线程被合流到配核（单核），libgomp barrier 风暴下性能崩塌
- 标记**秒级延迟生效**（内核周期评估，`affinity_period_ms=2000`）：短探针（10-300ms）测不到"将来会被合流"的状态——v7/v9 的探针全被这样骗过
- **「先试并行、不行退串行」必死**：试探本身触发标记，污染后连单线程都被降速 20-30×（实测污染后串行 0.6 GFLOPS vs 正常 23-35）。决策必须一次完成
- bench 自己的 B 生成阶段（38 个库内部线程跑秒级 dgemm）就可能预先触发标记——l_trsm 里无解，只能让损失有界
- **8 线程实测不触发标记**：掩码内闲核可借（真 8× 加速）；同一进程里 8 线程探针全速、38 线程正式工作瞬间灾难，是直接证据
- 标记会衰减：闲置 ~1 分钟后冷却（不同轮次表现不同的原因）

**对策**（TRSM v10，全部实现在 trsm.c）：
1. **正式工作封顶 8 线程**：即使中途被合流也只是 ~2.4× 串行的温和退化，灾难下界被结构性消灭（实测最坏 8.3s vs v6 的 95s）
2. 决策探针与正式工作**同构**（同线程数，测得即跑得）：
   - 信号一：sched_getcpu 同步采样（8 线程隔 20ms 各报所在 CPU，连续 5 点全同核 = 已被合流）→ 串行——直接观测机制本身，与天气/共租户忙闲无关
   - 信号二：8 线程纯 FMA 吞吐比 < 3（防配额限流容器：线程分散在不同核但总算力被压）→ 串行
3. 误选串行的代价（~6.4-7.9s 总计，仍领先榜首 ~2.5-3×）≪ 误选重线程并行的坍缩灾难，阈值一律偏向串行

### 17. 提交 zip 两个连环坑：反斜杠 + 子目录（2026-09-04，TRSM 提交两连失踪）

**现象**：TRSM 两次提交都不上榜单；同队 CONV/ZGEMM 包正常上榜。

**根因（两个独立致命点，先后踩中）**：
1. **反斜杠**：Windows PowerShell `Compress-Archive` 生成的 zip 条目分隔符是反斜杠（条目名字面为 `TRSM\trsm.c`，hex 5C）。Linux `unzip` 把 `\` 当文件名普通字符，解压出四个名字里带反斜杠的散件——没有目录、没有 run.sh，判题静默失败
2. **子目录**：修好反斜杠后（Linux 打包、正斜杠、小写 `trsm/` 目录）**依然不上榜**。对照齐思航成功上榜的 CONV 包发现：**文件必须直接位于 zip 根部**（无任何子目录）。判题系统解压后在根部找 run.sh，`trsm/run.sh` 找不到。官方文本"压缩zgemm目录为zip"实际含义是"压缩目录的**内容**"

**成功形态实拍**（齐思航 CONV 包，已在判题机跑通）：
```
zip 根部: bench_conv.c / conv2d.c / README.md / run.sh   ← 无子目录、无 exec 位（判题用 bash run.sh）
run.sh 要点: #!/bin/bash, set -euo pipefail, cd "$(dirname "$0")", 裸 gcc 无需 module
```

**解法**：
1. zip 在 Linux 上打：staging 目录内 `zip x.zip run.sh trsm.c bench_trsm.c README.md`（不带目录前缀）
2. 文件名模式照抄成功者：`trsm_optimized_latest_20260904.zip`（小写赛题名+描述+日期）
3. run.sh 无需依赖 exec 位（判题机 `bash run.sh`），但保留 `chmod +x` 无害；POSIX sh 兼容是廉价保险
4. run.sh: `cd "$(dirname "$0")"`（cwd 无关）+ 裸 gcc 先试 + module 双拼写 + find 兜底（搜索根含 /opt——dev 节点上 HPKit 装在 /opt/HPKit，含 sme/sve512/locking/nolocking 等 kblas 变体）
5. 判题全流程自测后再提交：干净目录解压 → 根部见 run.sh → `bash run.sh` 全用例 PASS

**验证手段**（PowerShell 检查 zip 条目）：
```powershell
$zip = [System.IO.Compression.ZipFile]::OpenRead("trsm.zip")
$zip.Entries | ForEach-Object { $_.FullName }   # 应为 run.sh / trsm.c / ...，直接在根、正斜杠
```
