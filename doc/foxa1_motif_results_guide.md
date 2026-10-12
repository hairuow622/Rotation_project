# FOXA1 motif 归因：每一步在做什么，结果怎么读

脚本在 `interpret/`，结果在 `results/interpret/FOXA1_A549_win240/`。下面两棵树按步骤标了每个文件是干什么的；四个 pattern 的子目录内容相同，只展开 pattern_0。`__pycache__/` 省略。

## 脚本：`interpret/`

```text
interpret/
├── prepare_seqs.py        # 1. 取 chr2 中心 240bp，打分，留下预测为 bound 的序列
├── attr_model.py          # 2. 把 Transformer 层改写成可归因的等价层
├── run_deeplift.py        # 3. DeepLiftShap，输出 hypothetical contribution
├── diagnose_attr.py       # 4. 检查守恒误差，并和 expected gradients 比较
├── annotate_patterns.py   # 6. 用 TOMTOM / JASPAR 给 pattern 归类
├── finemo_compat.py       # 7. 调用 Fi-NeMo 的兼容包装
└── summarize_motifs.py    # 8. 排名、位置分布、共现
```

MoDISco（第 5 步）和 Fi-NeMo 本身是外部命令，由 `submit_interpret.sh` 调用，不在这个目录里。

## 结果：`results/interpret/FOXA1_A549_win240/`

```text
FOXA1_A549_win240/
├── ohe.npz                      # 1. one-hot 序列 (N, 4, 240)
├── preds.tsv                    # 1. 每条序列的 logit、概率、是否保留
├── regions.bed                  # 1. 基因组坐标，summit = 120
├── hyp.npz                      # 3. hypothetical contribution，MoDISco 的输入
├── contrib.npz                  # 3. 真实碱基贡献 (hyp × one-hot)
├── diagnostics.tsv              # 4. 每条序列的守恒误差和相关性
├── diagnostics_summary.tsv      # 4. 原模型 vs 等价模型的汇总
├── diagnostics.png              # 4. 误差和相关性的直方图
├── modisco_results.h5           # 5. seqlet 和 4 个 pattern
├── modisco_report/              # 5. MoDISco + TOMTOM 报告
│   ├── motifs.html              #    每个 pattern 的 logo 和 JASPAR 匹配
│   ├── trimmed_logos/           #    pattern_0–3 的正反向 CWM
│   └── MA*.png                  #    匹配上的 JASPAR motif logo
├── patterns_annotation.tsv      # 6. pattern 的类别、共识序列、q 值
├── finemo_regions.npz           # 7. 带坐标的序列和贡献
├── finemo/                      # 7. 每次 hit 的原始结果
│   ├── hits.tsv                 #    每一次 hit：位置、链、贡献、相似度
│   ├── hits_unique.tsv          #    去掉重叠后的 hit
│   ├── hits.bed                 #    同上，BED 格式
│   ├── peaks_qc.tsv             #    每条 peak 的 hit 质控
│   ├── motif_data.tsv           #    每个 motif 的起止和缩放
│   ├── motif_cwms.npy           #    motif 的 CWM 矩阵
│   └── parameters.json          #    call-hits 的参数
├── finemo_report/               # 7. Fi-NeMo 报告
│   ├── report.html              #    总报告，浏览器打开
│   ├── motif_report.tsv         #    召回率、CWM 相似度、hit 数
│   ├── motif_occurrences.tsv    #    hit 出现情况
│   ├── seqlets.tsv              #    MoDISco seqlet 与 hit 的对应
│   ├── hit_vs_seqlet_counts.png #    hit 数和 seqlet 数的对比
│   ├── total_hit_distribution.png
│   ├── motif_cooocurrence.png   #    Fi-NeMo 自己的共现图（文件名少一个 c）
│   ├── motif_hit_distributions/ #    pattern_0–3 各自的 hit 数分布
│   └── CWMs/pos_patterns.pattern_0/   # pattern_1–3 结构相同
│       ├── modisco_fc.png/.txt        # MoDISco 正链 CWM
│       ├── modisco_rc.png/.txt        # MoDISco 反链 CWM
│       ├── hits_fc.png/.txt           # Fi-NeMo 全部 hit 的正链 CWM
│       ├── hits_rc.png/.txt           # Fi-NeMo 全部 hit 的反链 CWM
│       ├── hits_restricted_only.png/.txt  # 只在受限 hit 里出现的
│       └── seqlets_only.png/.txt      # 只在 seqlet 里、hit 没召回的
├── motif_ranking.tsv            # 8. 按贡献排序的汇总表
├── motif_ranking.png            # 8. 总贡献和序列覆盖率
├── motif_positions.png          # 8. hit 相对 peak 中心的位置
├── motif_cooccurrence.tsv       # 8. 观测共现 / 随机预期
└── motif_cooccurrence.png       # 8. 共现热图
```

这条分析回答的是：模型把一条 240 bp 序列判成 FOXA1 bound 时，主要是看哪些短序列。输入是 chr2 上 3218 个 peak 中心窗口；模型预测概率 ≥ 0.5 的 2629 条进入后续分析。方法细节见 [foxa1_motif_attribution.md](foxa1_motif_attribution.md)。

## 1. 准备序列（`prepare_seqs.py`）

取出 chr2 的 3218 个 peak，各取中心 240 bp，转成 one-hot，用已训练模型打分，只留下预测为 bound 的序列。

| 文件 | 内容 | 怎么读 |
|---|---|---|
| `ohe.npz` | 序列的 one-hot，形状 `(N, 4, 240)` | 后面所有步骤的序列输入 |
| `preds.tsv` | 每条的染色体坐标、logit、概率、是否保留 | `kept=True` 且 `prob≥0.5` 才进入归因；最终 2629 条，没有含 N 的序列 |
| `regions.bed` | 同一批区间，summit 设在第 120 bp | 给 Fi-NeMo 对齐基因组坐标用 |

## 2. 改写成可归因模型（`attr_model.py`）

原模型的 Transformer 层用函数式 attention，DeepLIFT 的规则挂不上去。这里按同样的计算顺序，用 `BilinearOp`、`Softmax`、`LayerNorm`、`ReLU` 重写了一层，权重直接从 checkpoint 拷过来，不重新训练。

这一步没有单独的结果文件。运行时用 512 条真实序列比对新旧模型，输出最大差约 `2e-6`，说明换层没有改预测。

## 3. 算每个碱基的贡献（`run_deeplift.py`）

对每条序列做 DeepLiftShap：拿 20 个 dinucleotide shuffle 当对照，问「这个位置换成某个碱基，会把 bound 的 logit 推高或压低多少」。正链和反向互补各算一次再平均。

| 文件 | 内容 | 怎么读 |
|---|---|---|
| `hyp.npz` | hypothetical contribution，形状 `(2629, 4, 240)` | 每个位置 4 个碱基各自「假如出现」的贡献。MoDISco 用的是这个 |
| `contrib.npz` | `hyp × one-hot` | 序列上真正出现的那个碱基的贡献，用来画 logo、做检查 |

正数推动 bound，负数抑制 bound。一条序列上所有真实碱基贡献加起来，应约等于「这条序列的预测 − 对照序列的预测」。

## 4. 检查分数能不能用（`diagnose_attr.py`）

在 300 条序列、每条 20 个对照（6000 对）上做两件事：贡献加总是否等于预测差（守恒），以及和另一种方法 expected gradients 是否指向同一批位置。

| 文件 | 内容 |
|---|---|
| `diagnostics_summary.tsv` | 两套模型的汇总数字 |
| `diagnostics.tsv` | 每条序列的明细 |
| `diagnostics.png` | 三张直方图 |

![attribution diagnostics](../images/foxa1_motif/diagnostics.png)

左图是守恒误差（对数坐标）：橙色（等价模型）堆在 `10⁻⁶` 附近，蓝色（原模型）堆在 `10⁰` 附近。中图是和 expected gradients 的 Pearson 相关，两种模型都集中在 0.9 左右。右图是贡献最高的前 20 个位置的重合度，中位数约 0.54。

| 模型 | 守恒误差中位数 | 与 expected gradients 的相关 | 能否用 |
|---|---|---|---|
| 原模型 | 0.53 | 0.92 | 否 |
| 等价模型 | 5.6×10⁻⁷ | 0.92 | 是 |

读法：原模型给出的「哪里重要」大体对，但分数加总不等于预测差，不能当贡献来加。等价模型的误差在浮点精度，后面所有「占总贡献百分之几」都建立在它上面。和 expected gradients 相关 0.92，说明重要位置是稳定的；前 20 个位置只重合一半，是因为两种方法对次要位置的排序不完全一样。

## 5. 把高贡献片段聚成 motif（TF-MoDISco）

在 2629 条序列的贡献图上找出重复出现的高贡献短片段（seqlet），再按相似度聚成 pattern。窗口必须设成 240，和序列等长。

| 文件 | 内容 | 怎么读 |
|---|---|---|
| `modisco_results.h5` | seqlet、pattern、CWM | 后面注释和 Fi-NeMo 的输入 |
| `/modisco_report/motifs.html` | 每个 pattern 的 logo，以及和 JASPAR 的比对 | 浏览器打开看序列长什么样、像哪个转录因子 |
| `/modisco_report/trimmed_logos/` | 裁剪后的正反向 logo | pattern 的序列图 |

结果是 3577 个正向 seqlet，聚成 4 个正向 pattern，没有负向 pattern。正向表示这些片段把预测往 bound 推；没有负向，表示模型里没有一组成簇、反复把预测往下拉的短序列。

## 6. 给 pattern 起名字（`annotate_patterns.py`）

用 TOMTOM 把每个 pattern 对到 JASPAR 2024。q < 0.05 的前 3 个匹配用来归类：Forkhead、共结合因子、复合 motif、低复杂度、未匹配。

文件是 `patterns_annotation.tsv`。

| pattern | 共识序列 | 类别 | 最佳匹配 | q 值 | seqlet 数 |
|---|---|---|---|---|---|
| pos_0 | TGTTTAC | Forkhead | FOXI1 / FOXA2 / FOXP1 | 5×10⁻⁴ | 2222 |
| pos_1 | TGACTCA | AP-1 共结合 | JUNB / BATF::JUN / BATF3 | 3.6×10⁻³ | 396 |
| pos_2 | TGTTTGCTTA | Forkhead 变体 | FOXD2 / FOXC2 / FOXE1 | 5.2×10⁻³ | 177 |
| pos_3 | GTAAATATTTGC | 未匹配 | 最接近 FOXA1，但不显著 | 0.59 | 46 |

q 值小才算匹配上。pos_3 的 0.59 不能当成「就是 FOXA1」。seqlet 数是 MoDISco 觉得「这个形状很强、反复出现」的次数，pos_0 占了绝大部分。

| pattern | CWM |
|---|---|
| pos_0 | ![pos_0](../images/foxa1_motif/pattern_0.cwm.fwd.png) |
| pos_1 | ![pos_1](../images/foxa1_motif/pattern_1.cwm.fwd.png) |
| pos_2 | ![pos_2](../images/foxa1_motif/pattern_2.cwm.fwd.png) |
| pos_3 | ![pos_3](../images/foxa1_motif/pattern_3.cwm.fwd.png) |

## 7. 在每条序列上定位这些 motif（Fi-NeMo）

MoDISco 只给出聚类原型。Fi-NeMo 拿这些原型回扫全部 2629 条序列，标出每次出现的位置、链、强度。

| 文件 | 内容 | 怎么读 |
|---|---|---|
| `finemo_regions.npz` | 带坐标的序列和贡献 | `call-hits` 的输入 |
| `/finemo/hits.tsv`、`/finemo/hits.bed` | 每一次 hit：染色体位置、链、`hit_importance`、相似度 | 一行一次出现。`hit_importance` 是这次 hit 对 bound 的贡献 |
| `/finemo_report/motif_report.tsv` | 每个 motif 的 hit 数、对 seqlet 的召回、CWM 相似度 | 用来判断扫描是否忠实于 MoDISco |
| `/finemo_report/report.html` | 上面这些的图 | 每个 pattern 有 MoDISco logo 和 Fi-NeMo hit logo 的对比 |

召回和相似度：

| pattern | seqlet 召回 | CWM 相似度 | hit 数 |
|---|---|---|---|
| pos_0 | 95% | 0.999 | 3051 |
| pos_1 | 98% | 0.995 | 1648 |
| pos_2 | 96% | 0.991 | 662 |
| pos_3 | 80% | 0.953 | 290 |

召回高、logo 相似度都在 0.95 以上，说明 Fi-NeMo 找到的就是 MoDISco 聚出来的那些形状。hit 数多于 seqlet 数是正常的：MoDISco 只收最强的片段来聚类，Fi-NeMo 会把同一条序列上较弱、但形状对得上的也标出来。AP-1 这一差距最大（396 个 seqlet，1648 个 hit）。

## 8. 汇总（`summarize_motifs.py`）

把上面的注释和 hit 收成三张最终图。

| 文件 | 内容 |
|---|---|
| `motif_ranking.tsv` / `motif_ranking.png` | 每个 motif 的覆盖率、平均贡献、总贡献、占全部正贡献的比例 |
| `motif_positions.png` | hit 相对 peak 中心的位置 |
| `motif_cooccurrence.tsv` / `motif_cooccurrence.png` | 两个 motif 是否比随机更常出现在同一条序列上 |

### 排名

![motif ranking](../images/foxa1_motif/motif_ranking.png)

左是总贡献，右是「2629 条 bound 序列里有多少条至少出现一次」。

| 排名 | motif | 有 hit 的序列 | 每个 hit 平均贡献 | 占全部正贡献 |
|---|---|---|---|---|
| 1 | Forkhead `TGTTTAC` | 88.1% | 2.44 | 44.5% |
| 2 | AP-1 `TGACTCA` | 38.6% | 1.21 | 11.9% |
| 3 | Forkhead 变体 `TGTTTGCTTA` | 21.4% | 1.75 | 6.9% |
| 4 | 未匹配 `GTAAATATTTGC` | 10.5% | 2.80 | 4.9% |

四个加起来约占全部正贡献的 68%。剩下约 32% 分散在没有聚成 pattern 的个别碱基上。

读法可以分成三层：

- Forkhead 是主信号。pos_0 和 pos_2 合计约 51%。88% 的 bound 序列至少有一个经典 `TGTTTAC`。匹配到的是 FOXI1、FOXA2、FOXP1 等，它们和 FOXA1 共用 Forkhead 核心，不是说峰上结合的是 FOXI1。
- AP-1 是最清楚的共结合 motif，出现在 39% 的序列里。单个 hit 的贡献大约是经典 Forkhead 的一半，所以它是常见的辅助信号，不是主信号。这和项目里「FOS-JUN 协助 FOXA1 结合」的方向一致。以前归因里出现过的 CEBPB，这次没有形成 pattern。
- pos_3 数量少，但单个 hit 平均贡献最高（2.80）。核心像是 Forkhead 的 `TGTTTAC` 把第 2 位 G 换成了 A，上游还有一段 `GTCAA`。JASPAR 对不上，把它当成非经典或复合位点，不能直接叫某个已知因子。

### 位置

![motif positions](../images/foxa1_motif/motif_positions.png)

横轴是相对 peak 中心的偏移，0 是 summit。四个 motif 都堆在中心大约 ±50 bp。Forkhead 最尖，AP-1 两侧拖得更开。模型用来判 bound 的位点就在 ChIP 峰中心附近，AP-1 可以离中心稍远。

### 共现

![motif co-occurrence](../images/foxa1_motif/motif_cooccurrence.png)

格子里是「观察到的共现 / 随机预期」。1 表示正好随机，大于 1 表示更常一起出现，小于 1 表示更少一起出现。四个配对都在 0.74 到 1.27，没有明显的必须成对或互相排斥。稍高的是 AP-1 和 Forkhead 变体 pos_2（1.27）；稍低的是两种 Forkhead 之间（0.74），多半是因为一条序列通常只会被收成其中一种 Forkhead 形式。
