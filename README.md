# Rotation Project

基于序列的 ChIP 结合预测与 domain adaptation（PyTorch Lightning）。

数据构建见 `dataset.py`：read count 训练集包含 bound shift、unbound GC-match / flanking / motif-match / random；domain discriminator 使用 accessible 与 motif-match 区域。

## 文件说明


| 文件                  | 作用                                                   |
| ------------------- | ---------------------------------------------------- |
| `dataset.py`        | 从 peak / genome 生成区域并写出 webdataset                   |
| `datamodule.py`     | Lightning DataModule，读取 webdataset                   |
| `models.py`         | 网络与训练逻辑（如 `ConvTowerDomain_v6`）                      |
| `data_config.yaml`  | 基因组、peak、染色体划分等路径                                    |
| `model_config.yaml` | Trainer / 模型超参                                       |
| `env.yaml`          | conda 环境导出                                           |
| `Makefile`          | 从 ENCODE 下载 FOXA1 conservative IDR thresholded peaks |
| `clean_peakdata.py` | 将 BED/narrowPeak 文件转换并排序为三列 BED                      |
| `submit_train.sh`   | 用 Slurm `sbatch` 提交 FOXA1 训练任务                      |




## 下载 FOXA1 peak 数据

在项目目录中运行：

```bash
make
```

Makefile 默认下载 ENCODE 文件 `ENCFF523NFH`，解压后生成：

```text
ENCSR000BRD_FOXA1_conservative_IDR_GRCh38.bed
```

如需指定其他 ENCODE file accession 或输出文件名：

```bash
make ACCESSION=ENCFF523NFH OUTPUT=custom_peaks.bed
```

删除 Makefile 生成的文件及临时文件：

```bash
make clean
```

运行 Makefile 需要系统中已安装 `curl` 和 `gzip`。

## 清理和排序 peak 数据

`clean_peakdata.py` 会读取 BED 或 narrowPeak 文件，仅保留染色体、起始坐标和结束坐标三列，然后按照自然染色体顺序（`chr1`–`chr22`、`chrX`、`chrY`、`chrM`）及区间坐标排序。

用法：

```bash
python clean_peakdata.py INPUT.bed OUTPUT.bed
```

处理 Makefile 下载的 FOXA1 peak 文件：

```bash
python clean_peakdata.py \
  ENCSR000BRD_FOXA1_conservative_IDR_GRCh38.bed \
  A549_FOXA1_CON_IDR_peaks.bed
```

输出是以 Tab 分隔的 BED3 文件，原始输入文件不会被修改。

## `dataset.py` 主要功能

该文件负责从 peak、基因组和可选的测序信号中生成训练、验证和测试数据。入口为 `python dataset.py -c data_config.yaml -o ... [--wds]`。

### 使用方法

运行前检查 `data_config.yaml` 中的输入路径，尤其是：

- `post_chip_peak`：BED3 格式的 ChIP-seq peak 文件
- `genome_fasta_file`：参考基因组 FASTA
- `genome_size_file`：参考基因组的 `.fai` 索引
- `blacklist_file`：需要排除的基因组区域
- `val_chrom` / `test_chrom`：验证集和测试集使用的染色体
- `generate_domain_data`：是否生成 `single_train_domain`；使用 `SingleDataModuleWds` 时设为 `false`

当前 A549 FOXA1 peak 配置应包含：

```yaml
post_chip_peak: /home/hvw5476/group/lab/hairuow/rotation_project/A549_FOXA1_CON_IDR_peaks.bed
generate_domain_data: false
```

仅生成划分和扩增后的 BED 文件：

```bash
python dataset.py \
  -c data_config.yaml \
  -o FOXA1_dataset_output
```

同时生成模型可读取的 WebDataset：

```bash
python dataset.py \
  -c data_config.yaml \
  -o FOXA1_dataset_output \
  --wds \
  -p 10
```

参数说明：

- `-c` / `--config`：输入 YAML 配置文件
- `-o` / `--output`：BED 和 WebDataset 的输出目录
- `--wds`：额外生成 WebDataset `.tar` 分片
- `-p`：写入 WebDataset 使用的并行进程数，默认为 10；应与申请到的 CPU 数量匹配
- `--normalizeBAM`：配置了 `post_chip_bam` 时，按每百万 mapped reads 对 BAM 信号归一化

上述命令在项目目录运行时，输出位于：

```text
FOXA1_dataset_output/
```

使用 `--wds` 时，每个 `.tar` 分片最多包含约 5000 个样本，并生成 `data_config.post_run.yaml`，其中记录产生的 BED 和 WebDataset 文件路径。


| 函数 / 类                               | 主要功能                                                                                              |
| ------------------------------------ | ------------------------------------------------------------------------------------------------- |
| `mark_region`                        | 检查每个区域是否与 ChIP-seq peak 重叠                                                                        |
| `remove_ambiguous_region`            | 删除容易混淆的负样本，例如中心没有 peak、但完整输入窗口碰到 peak 的区域                                                         |
| `define_training_coordinates`        | 生成模型的正负样本：正样本来自 peak 附近，负样本来自随机区域、相似 GC、相同 motif 和 peak 两侧                                        |
| `define_domain_task_coordinates_new` | 生成 domain adaptation 使用的区域，包括 accessible、inaccessible 和 motif-match 区域；没有 accessibility 数据时使用随机区域 |
| `define_random_coordinates`          | 从基因组中随机抽取区域，作为额外测试数据                                                                              |
| `genome_size_to_bdt`                 | 把染色体长度表转换成可供区间操作使用的 BedTool                                                                       |
| `define_coordinates_in_one_cell`     | 组织整个坐标生成过程：划分 train/val/test、扩增样本、添加正负链，并写出 BED 文件                                                |
| `writeWDS`                           | 从坐标中读取 DNA 序列和测序信号，保存为模型可直接读取的 WebDataset                                                         |
| `standardize_transform`              | 为 chromatin 信号和 ChIP target 设置标准化或归一化方法                                                           |
| `if __name__ == "__main__"`          | 读取配置并运行上述流程；使用 `--wds` 时还会写出 tar 文件和 `*.post_run.yaml`                                            |


主要数据流为：配置文件 + peak/基因组数据 → 生成正负样本坐标 → 划分 train/val/test → 写出 BED → 可选写成 WebDataset。

坐标区域长度是 `target_window_length`（256 bp）；写 WebDataset 时会在两侧补齐，使最终输入长度达到 1024 bp。

## `datamodule.py` 主要功能

该文件负责读取 `dataset.py` 生成的 WebDataset，并根据训练方式组织 DataLoader。当前 `model_config.yaml` 默认使用 `SingleDataModuleWds`。


| 函数 / 类                                                   | 主要功能                                                                           |
| -------------------------------------------------------- | ------------------------------------------------------------------------------ |
| `DataConfig`                                             | 读取包含 `webdataset` / `bed` 路径的 yaml；按数据集名称和样本类型查找文件，并构建 WebDataset pipeline     |
| `DataConfig.build_wds_pipeline`                          | 完成 shard 分配、样本打乱、解码和字段重命名，输出 `(seq, chrom, target, label)`；测试时可额外保留区域 key      |
| `MergedLoader` / `ChainedLoader` / `ChainedLoaderSample` | 分别用于合并两个 batch、顺序遍历多个 loader、随机采样多个 loader                                     |
| `collate_add_domain`                                     | 在普通 batch 末尾添加 source/target domain 标签；测试阶段据此区分数据来源                            |
| `SingleDataModuleWds`                                    | 单数据集训练：等概率混合 bound 与 unbound；分别创建 train、validation 和 test loader               |
| `MultiDataModuleWds`                                     | 多个 source 数据集联合训练：混合各数据集的正负样本，并串联验证集和测试集                                       |
| `DomainDataModuleWds`                                    | 同时提供 readcount source、domain source 和 domain target 数据，用于联合式 domain adaptation |
| `ADDADataModuleWds`                                      | 为 ADDA 提供 source/target domain loader，并用 target readcount 数据进行验证               |
| `ADDADataModuleWdsACC`                                   | 将 source/target 的 accessible 与 inaccessible 样本分开，构造四路 domain loader            |
| `center_and_expand_df` / `BedDataModule`                 | 将 BED 区域以中心扩展到固定窗口；不经过 WebDataset，直接从 BED、FASTA 和 bigWig 构建数据集                 |


主要数据流为：post-run yaml → `DataConfig` → WebDataset pipeline → Lightning DataModule → 模型所需 batch。

## `models.py` 主要功能

该文件同时定义网络组件、训练/验证/测试步骤、domain adaptation 模型变体和 LightningCLI 入口。当前配置使用 `ConvTowerDomain_v6`。


| 函数 / 类                                                          | 主要功能                                                                                       |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `TrainingRoutineHook`                                           | 通用 Lightning 训练基类：计算 readcount/分类损失和 domain 损失，并在测试结束后汇总预测、绘制 ROC/PRC 或回归散点图               |
| `GradientReversalFunction` / `GradientReversal`                 | 梯度反转层：前向传播保持输入不变，反向传播乘以负系数，使特征提取器学习 domain-invariant 表征                                    |
| `DenseBlock` / `ConvBlock` / `RConvBlock`                       | 全连接、卷积和残差卷积基础模块，用来搭建特征提取器与 domain discriminator                                            |
| `Squeeze` / `Residual` / `PositionalEncoding`                   | 张量维度压缩、残差连接和 Transformer 正弦位置编码等辅助模块                                                       |
| `ConvTowerDomain_v6`                                            | 主模型：DNA sequence（可选 chromatin）经 stem、卷积塔和 Transformer 后输出结合预测；实现当前单数据集的训练、验证和测试步骤          |
| `ConvTowerDomain_v6_New_PostAttn`                               | 简化 post-attention 层，并将 Transformer 特征展平后直接预测，减少原 post-attention 的正则化                       |
| `ConvTowerDomain_v6_GradientReversal_SplitSeqChrom_MultiDomain` | 在 sequence/chromatin 卷积特征后分别接梯度反转和多分类 domain predictor，进行多域联合训练                            |
| `ConvTowerDomain_v6_ADDA`                                       | 基于预训练 checkpoint 执行 ADDA；冻结 source 编码器和分类器，对抗训练 target chromatin 编码器与 domain discriminator |
| `ConvTowerDomain_v6_ADDA_ACC`                                   | ADDA 四分类变体，显式区分 source/target 与 accessible/inaccessible 四类 domain                          |
| `ConvTowerDomain_v6_ADDA_SeqChrom`                              | 可分别控制 sequence 和 chromatin 分支是否执行 ADDA，并为两个分支配置独立 discriminator                            |
| `construct_domain_attention_discriminator`                      | 构建由位置编码、Transformer 和线性输出层组成的 domain discriminator                                         |
| `cli_main`                                                      | 启动 LightningCLI，装配模型、DataModule、checkpoint、模型摘要和学习率监控回调                                    |


主模型数据流为：`seq/chrom` → stem → convolutional tower → positional encoding + Transformer → post-attention → prediction。

## 使用 Weights & Biases 记录训练

`model_config.yaml` 已配置 `WandbLogger`，无需修改 `models.py`。模型中通过 `self.log(...)` 记录的训练损失、验证损失和准确率会自动同步到 W&B。

首次配置环境时安装兼容新版 API key 的客户端并登录：

```bash
pip install "wandb==0.22.3"
wandb login --relogin
```

不要把 API key 写入配置文件或提交到 Git。当前日志会上传至：

```text
https://wandb.ai/hairuow-carnegie-mellon-university/Rotation_project
```

使用生成 WebDataset 时得到的 `data_config.post_run.yaml` 启动训练：

```bash
python models.py fit \
  --config model_config.yaml \
  --data.config_file data_config.post_run.yaml
```

默认 run 名称是 `FOXA1_ConvTowerDomain_v6`。可以在命令行中为每次实验指定唯一名称：

```bash
python models.py fit \
  --config model_config.yaml \
  --data.config_file data_config.post_run.yaml \
  --trainer.logger.init_args.name FOXA1_run_001
```

W&B 会记录：

- `model_config.yaml` 中的模型超参数
- `train_readcount_entropy_loss`
- `train_accuracy` / `val_accuracy`
- `val_loss`
- learning rate、epoch 和 global step
- 最优及最终 checkpoint（`log_model: true`）

集群计算节点无法联网时，可先离线记录：

```bash
WANDB_MODE=offline python models.py fit \
  --config model_config.yaml \
  --data.config_file data_config.post_run.yaml
```

任务结束后，在能联网的节点同步本地 run：

```bash
wandb sync wandb/offline-run-*
```

## 查看 TensorBoard 日志（`tb_logs`）

训练或测试若使用 `TensorBoardLogger`，事件文件会写在 `tb_logs/<run_name>/version_*` 下。当前环境的 TensorBoard 与 protobuf 存在兼容问题，**每次**在新终端启动 TensorBoard 前需设置：

```bash
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

在 **Cursor 已 SSH 连接的主机**（或登录节点）上查看即可；日志在共享盘上，不必在分配 GPU 的计算节点上开 TensorBoard。进入项目目录并激活环境：

```bash
cd /home/hvw5476/group/lab/hairuow/rotation_project
conda activate rotation
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
tensorboard --logdir tb_logs/FOXA1_run_002_test/version_1 --port 6006 --host 127.0.0.1
```

将 `--logdir` 换成其它路径即可查看不同 run，例如整个 `tb_logs`、某一 run 的全部 version：

```bash
tensorboard --logdir tb_logs --port 6006 --host 127.0.0.1
tensorboard --logdir tb_logs/FOXA1_run_002_test --port 6006 --host 127.0.0.1
```

在浏览器中打开：

1. Cursor **Ports（端口）** 面板中转发 `6006`，点击 **Open in Browser**；或
2. 本机浏览器访问 **http://localhost:6006**（需 Cursor 或 SSH 已将远程 `127.0.0.1:6006` 映射到本机）。

若 6006 已被占用，可改用其它端口（如 `6007`），并在 Ports / SSH 中转发对应端口。

不建议在计算节点上使用 `--bind_all` 后访问终端里显示的 `http://E1-xxxxx.cm.cluster:6006/`：笔记本通常无法直连计算节点，应使用 `--host 127.0.0.1` 配合端口转发。

看完后在运行 TensorBoard 的终端按 `Ctrl+C` 结束进程。

## 使用最优 checkpoint 运行测试

在已分配 GPU 的计算节点上，进入项目目录并激活环境：

```bash
cd /home/hvw5476/group/lab/hairuow/rotation_project
conda activate rotation
```

若测试时使用 TensorBoard logger，请在同一终端设置 `export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`（原因与用法见上文「查看 TensorBoard 日志」）。

使用 `FOXA1_run_002` 的最优 checkpoint（epoch 20，val_loss 0.361897）运行测试：

```bash
python models.py test \
  --config model_config.yaml \
  --data.config_file data_config.post_run.yaml \
  --ckpt_path checkpoints/FOXA1_run_002/best-epoch=20-val_loss=0.361897.ckpt \
  --trainer.devices 1 \
  --trainer.strategy auto \
  --trainer.logger.init_args.name FOXA1_run_002_test
```

上述命令使用 `model_config.yaml` 中的 W&B logger，测试 run 名称为 `FOXA1_run_002_test`。测试使用单个设备，加载指定权重，不重新训练。

分类测试按 dataloader 记录 auROC、auPRC 和正类曲线：W&B 使用 `wandb.plot.roc_curve()`、`wandb.plot.pr_curve()`；配置为 TensorBoard logger 时使用 `add_figure()`。预测结果保存为 `predictions.txt`，具体路径会在终端输出。

## 使用 `sbatch` 提交训练

`submit_train.sh` 会在 `mahony` 分区申请 1 张 GPU、4 个 CPU、64G 内存和 24 小时，并在 `rotation` 环境中运行上述 `models.py fit` 命令。

提交任务：

```bash
cd /home/hvw5476/group/lab/hairuow/rotation_project
sbatch submit_train.sh
```

默认 `RUN_NAME` 为 `FOXA1_YYYYMMDD_HHMMSS`。该值同时用于本地 checkpoint 目录 `checkpoints/${RUN_NAME}/` 和 W&B 的 `--trainer.logger.init_args.name`，避免同名覆盖。

自定义名称：

```bash
RUN_NAME=FOXA1_run_001 sbatch submit_train.sh
```

查看任务和日志：

```bash
squeue -u $USER
tail -f logs/FOXA1_train_<jobid>.out
```

`sbatch` 会直接读取脚本内容，**不要求可执行权限**。仓库里已经执行过 `chmod +x submit_train.sh`，这只是方便你写成 `./submit_train.sh` 在本地调试；提交作业时用 `sbatch submit_train.sh` 即可。不要用 `chmod 777`。

计算节点需能读取已登录的 W&B 凭据（通常在登录节点先运行 `wandb login`）。若计算节点无法联网，在脚本中设置 `WANDB_MODE=offline`，结束后再 `wandb sync`。

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