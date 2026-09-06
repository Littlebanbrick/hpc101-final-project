# chores/ — 过程产物

本目录存放不进入最终提交结构的中间产物。提交版根目录必须保持大作业文档要求的
干净布局（`CONV/`、`TRSM/`、`ZGEMM/`、`report.pdf`、`presentation.pptx`），
所有「写作 / 调试 / 数据」类文件都收在这里。

## 子目录约定

| 目录 | 用途 |
| --- | --- |
| `report/` | 实验报告源码（Typst）与图表素材；编译产物 `report.pdf` 输出到仓库根目录 |
| `bench/` | 鲁鹏集群上跑出的基准测试日志、CSV，以及生成加速比表/图的脚本 |

## 报告编译

```bash
cd chores/report
typst compile report.typ ../../report.pdf    # 输出到根目录以满足提交结构
```

或开发期增量预览：

```bash
typst watch chores/report/report.typ report.pdf
```
