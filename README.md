# Rotation Project

基于序列的 ChIP 结合预测与 domain adaptation（PyTorch Lightning）。

数据构建见 `dataset.py`：read count 训练集包含 bound shift、unbound GC-match / flanking / motif-match / random；domain discriminator 使用 accessible 与 motif-match 区域。

## 文件说明

| 文件 | 作用 |
|------|------|
| `dataset.py` | 从 peak / genome 生成区域并写出 webdataset |
| `datamodule.py` | Lightning DataModule，读取 webdataset |
| `models.py` | 网络与训练逻辑（如 `ConvTowerDomain_v6`） |
| `data_config.yaml` | 基因组、peak、染色体划分等路径 |
| `model_config.yaml` | Trainer / 模型超参 |
| `env.yaml` | conda 环境导出 |

## 环境

```bash
conda env create -f env.yaml
conda activate torch_may_2025
```

## 数据路径

`data_config.yaml` 中的基因组、blacklist、peak 使用集群绝对路径，例如：

- peak: `/data/lab/hairuow/K562_CTCF_CON_IDR_peaks.bed`
- genome: `hg38.fa` / `hg38.fa.fai`
- blacklist: `hg38-blacklist.v2.bed`

请按本机路径修改后再跑。大数据文件（bed/bam/bigwig/fasta/webdataset）已由 `.gitignore` 排除。

## 配置要点

- 输入窗口 1024 bp，target 窗口 256 bp
- 验证集：`chr3`；测试集：`chr2`
- 模型：`ConvTowerDomain_v6`，`seqonly: True`，分类任务

## 仓库范围

本仓库只版本管理代码与配置，不包含原始测序数据和训练产物。
