# ChIP-ISO project

Collaboration with Cheng Xu from Lucy lab, they developed an experiment method ChIP-ISO that is able to measure TF binding strength to designed oligos at certain locus in vivo.

My part in this project is to use the Bichrom model to validate their experiment findings, including:

1. FOS-JUN can assist FOXA1 binding

2. H3K9me3 can't help FOXA1 binding

## Model training, evaluation, and feature attribution in A549

### training

> workdir: 2022/Jun13_Binary_Bichrom_on_genome_wide_chip

I trained the Bichrom model by using DNA sequence and H3K9me3 to predict FOXA1 binding, FOXA1 ChIP-seq peaks and H3K9me3 profile bigwig file are from Lucy lab. I downloaded ATAC-seq data in A549 from ENCODE (file id: ENCFF876UEM). Check `main.sh` for the commands.

After training the model, the performance on held out test chromosome is:
```bash
#2022/Jun13_Binary_Bichrom_on_genome_wide_chip/bichrom_out/A549_foxa1_chip_H3K9me3_win240/step2_output/metrics.txt
Model:MSEQ
AUC ROC:0.917852623949172
AUC PRC:0.24275437036980002
Model:MSC
AUC ROC:0.9187241631265979
AUC PRC:0.21652207400136744
```

### evaluation vs ISO signal

Then I tested the seq-only model's performance on ISO library sequences, this is to double check the consistency between the model and ChIP-ISO experiments. Check `2022/Jun13_Binary_Bichrom_on_genome_wide_chip/Jun13_Binary_Bichrom_on_genome_wide_chip.ipynb` for detailed steps. The performance is:

![Bichrom predict vs ISO signal histogram](2022/Jun13_Binary_Bichrom_on_genome_wide_chip/ChIP-ISO_bichrom_predict_vs_ISO_signal.png)

![Bichrom predict vs ISO signal barplot](2022/Jun13_Binary_Bichrom_on_genome_wide_chip/ChIP-ISO_bichrom_predict_vs_ISO_signal_barplot.png)

### feature attribution

> workdir: 2022/Jun13_Binary_Bichrom_on_genome_wide_chip

Fimo scanning results on ISO sequences are from `Jun27_FIMO_motif_scan_on_ISO_sequence`

I used DeepLiftSHAP implementation (DeepExplainer) from [SHAP repo](https://github.com/slundberg/shap) to interpret the prediction on ISO library sequences. The control sequences are sampled from di-nucleotide shuffled version of ISO library sequences (200 sequences in total). The attribution scores are visualized as below:

![feature attribution ori](2022/Jun13_Binary_Bichrom_on_genome_wide_chip/feature_attr_ori_heatmap_FOXA1_JUN_CEBPB.png)

### TF-Modisco

> workdir: 2022/Dec12_TF-Modisco_on_binary_bichrom_on_ISO_library

> Note the shap version used for this analysis is my fork: https://github.com/yztxwd/shap.git, this is to customize the combine function of DeepExpainer according to https://docs.google.com/presentation/d/1JCLMTW7ppA3Oaz9YA2ldDgx8ItW9XHASXM1B3regxPw/edit?pli=1#slide=id.g65244ca07a_0_243

Global DNA feature attribution by TF-Modisco gave the results on sequence+H3K9me3 bichrom model:

![Modisco on H3K9me3 Bichrom](2022/Dec12_TF-Modisco_on_binary_bichrom_on_ISO_library/modisco_reports/model_H3K9me3_FoxA1_ChIP_A549_MACS2_callpeak_narrowPeak_240bp_around_summit_vs_dinuc_shuffle_on_cisBP_hypothetical_on_preactivation_swapCT_v1_modisco_report/motifs.pdf)


### seq-only vs seq+chrom in repressive regions

> workdir: 2022/Aug01_Seq-only_vs_Bichrom_in_repressive_regions

Compare the performance of seq-only and seq+chrom models in repressive regions specifically. The purpose is to check if H3K9me3 would be helpful to FOXA1 binding predicton in closed chromatin. Repressive regions are from Cheng.

![seq-only vs bichrom in repressive regions](2022/Aug01_Seq-only_vs_Bichrom_in_repressive_regions/auPRC_seq_vs_bichrom.png)

### seq-only vs seq+H3K9me3 vs seq+ATAC vs seq+ATAC+All Histone modifications in repressive/all test regions

> workdir: 2023/Sep22_Compare_Bichrom_model_under_various_settings

![all models in repressive regions](2023/Sep22_Compare_Bichrom_model_under_various_settings/figures/Native_plus_his_mk_TFChIP_Epi_FoxA1_extend_ud20bp_auPRC_all_models.pdf)

![all models in test regions](2023/Sep22_Compare_Bichrom_model_under_various_settings/figures/bichrom_out_H3K9me3_data_test_auPRC_all_models.pdf)

## DNA Shape analysis

> workdir: 2022/Nov21_DNAShapeR_on_EMSAseq_library
> EMSAseq library should be a more comprehensive sequence library than ISO library in Aug07_DNAShapeR_on_ISO_library/

I think we were exploring alternative explanation for FOXA1 binding preference to certain sequences, so I checked the DNA shape to see if it is associated with ISO signals.

Sequences are grouped into bins and plotted as below:

![DNA shape merged](2022/Nov21_DNAShapeR_on_EMSAseq_library/DNAshapeR_grid_merge_plot.png)

![DNA shape separate](2022/Nov21_DNAShapeR_on_EMSAseq_library/DNAshapeR_grid_plot.png)

## Train Bichrom on FOXA1 in multiple cell lines and interpret

Train FOXA1 bichrom model in more cell lines, and interpret the global DNA sequence motifs

The datasets are from two sources:

1. ENCODE: HepG2, K562, MCF7
2. GEO: 22Rv1, Bladder_Cancer_Basal, Bladder_Cancer_Luminal, C42B, Cyt49_HP, Cyt49_PP2, GP5d, Huh7, LNCaP, NCI_H441, RT4, SKBR3, T47D

The reports are in `2023/Mar18_Bichrom_on_multiple_cell_types/modisco_reports`