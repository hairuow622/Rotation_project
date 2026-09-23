Description="NN model collection"

from copy import deepcopy
import inspect
import os
import sys

from einops.layers.torch import Rearrange
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.cli import LightningCLI
from scipy.stats import pearsonr
import seaborn as sns
import torch
from torch import nn, optim
from torch.autograd import Function
import torch.nn.functional as F
from torch.optim.lr_scheduler import ChainedScheduler, ExponentialLR, LambdaLR, CyclicLR, MultiplicativeLR, OneCycleLR, ReduceLROnPlateau, SequentialLR, StepLR
from torchmetrics import MeanSquaredError, MetricCollection, PearsonCorrCoef, Accuracy
from torchmetrics.classification import BinaryROC, BinaryPrecisionRecallCurve
from sklearn import metrics

torch.set_float32_matmul_precision('medium')

class TrainingRoutineHook(pl.LightningModule):
    """
    Define universal things:
    1. Dataloader
    2. Hooks
    3. training/val/test steps
    """
    def __init__(self):
        super().__init__()
        self.accuracy = Accuracy(task="binary")

        self.test_step_outputs = []
    
    def training_step(self, batch, batch_idx):
        # define train loop
        seq_r_s_b, chroms_r_s_b, y_r_s_b, label_r_s_b = batch['readcount_source_bound']
        seq_r_s_ub, chroms_r_s_ub, y_r_s_ub, label_r_s_ub = batch['readcount_source_unbound']
        seq_d_s, chroms_d_s, y_d_s, label_d_s, domain_d_s = batch['domain_source']
        seq_d_t, chroms_d_t, y_d_t, label_d_t, domain_d_t = batch['domain_target']
        
        # merge inputs
        seq_merge = torch.cat((seq_r_s_b, seq_r_s_ub, seq_d_s, seq_d_t))
        chroms_merge = torch.cat((chroms_r_s_b, chroms_r_s_ub, chroms_d_s, chroms_d_t))
        
        pred_target_merge, pred_domain_merge = self(seq_merge, chroms_merge)
        r_s_size = seq_r_s_b.shape[0] + seq_r_s_ub.shape[0]
        pred_target_r_s = pred_target_merge[:r_s_size]
        pred_domain_d = pred_domain_merge[r_s_size:]

        # compute prediction and loss
        if self.classification:
            target_loss = F.binary_cross_entropy_with_logits(pred_target_r_s, torch.cat((label_r_s_b, label_r_s_ub)).float())
            accuracy = self.accuracy(F.sigmoid(pred_target_r_s), torch.cat((label_r_s_b, label_r_s_ub)))
            self.log('train_readcount_entropy_loss', target_loss)
            self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        else:
            target_loss = F.mse_loss(pred_target_r_s, torch.cat((y_r_s_b, y_r_s_ub)))
            self.log('train_readcount_MSE_loss', target_loss)
        domain_y = torch.cat((domain_d_s, domain_d_t))
        domain_loss = F.binary_cross_entropy_with_logits(pred_domain_d, domain_y)
        self.log('train_domain_entropy_loss', domain_loss)
        return {'loss': target_loss + domain_loss}

    def training_step_no_domain(self, batch, batch_idx):
        # define train loop
        seq_r_s_b, chroms_r_s_b, y_r_s_b, label_r_s_b = batch['readcount_source_bound']
        seq_r_s_ub, chroms_r_s_ub, y_r_s_ub, label_r_s_ub = batch['readcount_source_unbound']
        
        # merge inputs
        seq_merge = torch.cat((seq_r_s_b, seq_r_s_ub))
        chroms_merge = torch.cat((chroms_r_s_b, chroms_r_s_ub))

        pred_target = self(seq_merge, chroms_merge)
        if type(pred_target) == tuple:
            pred_target_r_s = pred_target[0]
        else:
            pred_target_r_s =  pred_target

        # compute prediction and loss
        if self.classification:
            target_loss = F.binary_cross_entropy_with_logits(pred_target_r_s, torch.cat((label_r_s_b, label_r_s_ub)))
            accuracy = self.accuracy(F.sigmoid(pred_target_r_s), torch.cat((label_r_s_b, label_r_s_ub)))
            self.log('train_readcount_entropy_loss', target_loss)
            self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        else:
            target_loss = F.mse_loss(pred_target_r_s, torch.cat((y_r_s_b, y_r_s_ub)))
            self.log('train_readcount_MSE_loss', target_loss)
        return {'loss': target_loss}

    def validation_step(self, batch, batch_idx):
        # define validation loop
        seq, chroms, y, label = batch
        y_target, y_domain = self(seq, chroms)

        # compute prediction and loss
        if self.classification:
            val_loss = F.binary_cross_entropy_with_logits(y_target, label)
            accuracy = self.accuracy(F.sigmoid(y_target), label)
            self.log('val_loss', val_loss, sync_dist=True)
            self.log('val_accuracy', accuracy, sync_dist=True, on_epoch=True, prog_bar=True)
        else:    
            val_loss = F.mse_loss(y_target, y)
            self.log('val_loss', val_loss, sync_dist=True)
        return {'val_loss': val_loss}
    
    # Using custom or multiple metrics (default_hp_metric=False)
    def on_test_start(self):
        # TensorBoard log_hyperparams accepts a metrics placeholder; WandbLogger only accepts params.
        metrics = {"hp/auROC": 0, "hp/auPRC": 0, "hp/MSE": 0, "hp/PearsonR": 0}
        logger = self.logger
        if hasattr(logger, "loggers"):
            loggers = logger.loggers
        else:
            loggers = [logger]
        for lg in loggers:
            params = inspect.signature(lg.log_hyperparams).parameters
            if "metrics" in params:
                lg.log_hyperparams(self.hparams, metrics)
            else:
                lg.log_hyperparams(self.hparams)
        
    def on_test_epoch_start(self):
        # ensure world size is 1
        if self.trainer.world_size != 1:
            print(f"World size is {self.trainer.world_size}")
            print(f"Please set # of devices as 1, distributed strategy on multiple devices could lead to incorrect prediction tensor shape")
            sys.exit(1)
        
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # define test
        key, seq, chroms, y, label, domain = batch
        pred_target, pred_domain = self(seq, chroms)

        # compute prediction and loss
        if self.classification:
            test_loss = F.binary_cross_entropy_with_logits(pred_target, label)
            self.test_step_outputs.append({'key': key, 'pred': pred_target, 'true': label, 'domain': domain, 'dataloader_idx': torch.Tensor([dataloader_idx]).repeat(seq.shape[0])})
        else:
            test_loss = F.mse_loss(pred_target, y)
            self.test_step_outputs.append({'key': key, 'pred': pred_target, 'true': y, 'label': label, 'domain': domain, 'dataloader_idx': torch.Tensor([dataloader_idx]).repeat(seq.shape[0])})

        return test_loss
    
    def on_test_epoch_end(self):
        # collect outputs from each batch
        out_keys = []
        out_preds = []
        out_trues = []
        out_labels = []
        out_domains = []
        out_dataloader_idx = []
        for test_step_loader_outputs in self.test_step_outputs:
            out_keys.append(test_step_loader_outputs['key'])
            out_preds.append(test_step_loader_outputs['pred'])
            out_trues.append(test_step_loader_outputs['true'])
            if not self.classification: out_labels.append(test_step_loader_outputs['label'])
            out_domains.append(test_step_loader_outputs['domain'])
            out_dataloader_idx.append(test_step_loader_outputs['dataloader_idx'])
        
        out_keys = np.concatenate(out_keys)
        out_preds = torch.cat(out_preds).detach().cpu().flatten()
        out_trues = torch.cat(out_trues).detach().cpu().flatten()
        out_domains = torch.cat(out_domains).detach().cpu().flatten()
        out_dataloader_idx = torch.cat(out_dataloader_idx).detach().cpu().flatten()
        if not self.classification: out_labels = torch.cat(out_labels).detach().cpu().flatten()
               
        if self.classification:
            out_preds_sig = F.sigmoid(out_preds.float()) # Converted everything to float to avoid  ScalarType BFloat16
            out_trues = out_trues.float()
            out_domains = out_domains.float()
            out_dataloader_idx = out_dataloader_idx.float()
        
            df_save = pd.DataFrame({"Region": out_keys, "Truth": out_trues, "Predictions": out_preds_sig, "Domain": out_domains, "Dataloader_idx": out_dataloader_idx})
            print(f"entropy loss: {F.binary_cross_entropy_with_logits(out_preds, out_trues)}")
            print(f"accuracy: {self.accuracy(out_preds_sig, out_trues)}")
            
            ## ROC and PRC curves
            fig_roc, ax_roc = plt.subplots()
            fig_prc, ax_prc = plt.subplots()
            for idx in np.unique(out_dataloader_idx):
                out_trues_sub = out_trues[out_dataloader_idx==idx]
                out_preds_sig_sub = out_preds_sig[out_dataloader_idx==idx]

                display = metrics.RocCurveDisplay.from_predictions(out_trues_sub, out_preds_sig_sub)
                auROC = metrics.roc_auc_score(out_trues_sub, out_preds_sig_sub)
                display.plot(ax=ax_roc, label=f'Dataloader {idx}; auROC={auROC:.2f}')
                
                display = metrics.PrecisionRecallDisplay.from_predictions(out_trues_sub, out_preds_sig_sub)
                auPRC = metrics.average_precision_score(out_trues_sub, out_preds_sig_sub)
                display.plot(ax=ax_prc, label=f'Dataloader {idx}; auPRC={auPRC:.2f}')
                
                self.log(f"hp/auROC_{idx}", auROC, sync_dist=True)
                self.log(f"hp/auPRC_{idx}", auPRC, sync_dist=True)
            self.logger.experiment.add_figure("ROC on test data", fig_roc)
            self.logger.experiment.add_figure("PRC on test data", fig_prc)
        else:
            out_preds = out_preds.float()  # Converted everything to float to avoid  ScalarType BFloat16
            out_trues = out_trues.float()
            out_labels = out_labels.float()
            out_domains = out_domains.float()
            out_dataloader_idx = out_dataloader_idx.float()        
            df_save = pd.DataFrame({"Region": out_keys, "Truth": out_trues, "Predictions": out_preds, "Label": out_labels, "Domain": out_domains, "Dataloader_idx": out_dataloader_idx})
            ## scatterplot on all test data
            axis_limit = max(np.percentile(out_preds, 99), np.percentile(out_trues, 99))
            fig = plt.figure(figsize=(12, 12))
            jg = sns.jointplot(x='Predictions', y='Truth', hue='Label', palette=['orange', 'deepskyblue'],
                               data=df_save, alpha=0.05)
            jg.ax_joint.axline((0, 0), slope=1, linestyle='--', color='black')
            jg.ax_joint.set_xlim(left=0, right=axis_limit)
            jg.ax_joint.set_ylim(bottom=0, top=axis_limit)
            jg.ax_joint.set_xlabel("Predictions by Model")
            jg.ax_joint.set_ylabel("True target")
            jg.ax_joint.text(0.1, 0.8, f"pearsonr correlation efficient/p-value \n{pearsonr(out_preds, out_trues)[0]:.2f}", transform=plt.gca().transAxes, fontsize='large')
            jg.ax_joint.text(0.1, 0.7, f"mean suqare error \n{np.square(out_preds-out_trues).mean():.4f}", transform=plt.gca().transAxes, fontsize='large')
            self.logger.experiment.add_figure(f"Prediction vs True on whole test dataset", jg.figure)
            self.log("hp/MSE", metrics.mean_squared_error(out_trues, out_preds))
            self.log("hp/PearsonR", pearsonr(out_trues, out_preds)[0])

        df_save.to_csv(os.path.join(self.logger.log_dir, "predictions.txt"), header=True, index=False, sep="\t")

class squeeze(nn.Module):
    def forward(self, x):
        return torch.squeeze(x)
    
# https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/revgrad.py
class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1.):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

def default(val, d):
    return val if val is not None else d

def exponential_linspace_int(start, end, num, divisible_by = 1):
    def _round(x):
        return int(round(x / divisible_by) * divisible_by)

    base = math.exp(math.log(end / start) / (num - 1))
    return [_round(start * base**i) for i in range(num)]

def DenseBlock(num_f, activation=nn.LeakyReLU, dropout=0.):
    return nn.Sequential(
        nn.Linear(num_f, num_f),
        activation(),
        nn.BatchNorm1d(num_f),
        nn.Dropout(dropout)
    )

def ConvBlock(dim, dim_out=None, kernel_size = 1, stride = 1, dilation=1):
    "Standard convolutional block"
    return nn.Sequential(
        nn.Conv1d(dim, default(dim_out, dim), kernel_size, padding = int(dilation*(kernel_size-1)/2), stride=stride, dilation=dilation),
        nn.GELU(),
        nn.BatchNorm1d(default(dim_out, dim)),
    )

def RConvBlock(dim, kernel_size = 1, num_conv=1, skip=True):
    "A convolutional block stack with skip connnection"
    layer_list = []
    for i in range(num_conv): 
        layer_list.extend([
            nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2),
            nn.GELU(),
            nn.BatchNorm1d(dim),
        ])
    if skip:
        return Residual(nn.Sequential(*layer_list))
    else:
        return nn.Sequential(*layer_list)

class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim is None:
            return torch.squeeze(x)
        else:
            return torch.squeeze(x, dim=self.dim)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ConvTowerDomain_v6(TrainingRoutineHook):
    """
    Convolutional tower model v6
    Separate sequence and chromatin convolutional tower for only applying domain adaptation to one of them
    Final Attention layers merge seq and chrom
    """
    def __init__(self,
                 chroms_channel=1,
                 input_len=1024,
                 conv1d_filter=64,
                 conv_tower_depth=4,
                 conv_tower_kernel=5,
                 conv_tower_dilation=1,
                 attn_num_layers=2,
                 attn_num_heads=8,
                 attn_dim_feedforward=2048,
                 activation="LeakyReLU",
                 dropout=0.,
                 gamma=0.99995,
                 lr=1e-4,
                 lambd=0,
                 classification=False,
                 seqonly=False):
        super().__init__()
        self.chroms_channel = chroms_channel
        self.input_len = input_len
        self.conv1d_filter = conv1d_filter
        self.conv_tower_depth = conv_tower_depth 
        self.conv_tower_kernel = conv_tower_kernel
        self.conv_tower_dilation = conv_tower_dilation
        self.attn_num_layers = attn_num_layers
        self.attn_num_heads = attn_num_heads
        self.attn_dim_feedforward = attn_dim_feedforward
        self.activation = getattr(nn, activation)
        self.dropout = dropout
        self.gamma = gamma
        self.lr = lr
        self.lambd = lambd
        self.classification = classification
        self.seqonly = seqonly
        self.save_hyperparameters()
        
        self.example_input_array = (torch.zeros(512, 4, self.input_len).index_fill_(1, torch.tensor(2), 1), torch.ones(512, self.chroms_channel, self.input_len))
        
        # stem
        self.seq_stem = nn.Sequential(
            nn.Conv1d(4, self.conv1d_filter, 25, padding=12, bias=False),
            self.activation(),
            nn.BatchNorm1d(self.conv1d_filter)            
        )
        if not self.seqonly:
            self.chrom_stem = nn.Sequential(
                nn.Conv1d(self.chroms_channel, self.conv1d_filter, 25, padding=12, bias=False),
                self.activation(),
                nn.BatchNorm1d(self.conv1d_filter)            
            )

        # convolutional tower
        ## compute the depth because the exponential base is fixed to 2 here
        self.conv_tower_outdim = int(self.conv1d_filter * (2**int(self.conv_tower_depth-1)))
        self.conv_tower_outlen = int(self.input_len / (2**int(self.conv_tower_depth-1)))
        dim_lists = exponential_linspace_int(self.conv1d_filter, self.conv_tower_outdim, num=self.conv_tower_depth,  divisible_by=2)
        conv_tower = []
        for dim_in, dim_out in zip(dim_lists[:-1], dim_lists[1:]):
            conv_tower.append(nn.Sequential(
                    RConvBlock(dim_in, kernel_size=self.conv_tower_kernel, skip=True),
                    ConvBlock(dim_in, dim_out=dim_out, kernel_size=1),
                    nn.MaxPool1d(kernel_size=2)
                )
            )
        self.seq_conv_tower = nn.Sequential(*conv_tower)
        if not self.seqonly: 
            self.chrom_conv_tower = deepcopy(self.seq_conv_tower)
        merge_dim = self.conv_tower_outdim if self.seqonly else self.conv_tower_outdim * 2
        
        # self attention
        self.pos_encoder = PositionalEncoding(merge_dim, self.dropout, self.conv_tower_outlen) # assume we merged sequence and chromatin inputs here
        transformer_encoder = []
        for i in range(self.attn_num_layers):
            transformer_encoder.append(nn.TransformerEncoderLayer(d_model=merge_dim,
                                                                  nhead=self.attn_num_heads,
                                                                  dim_feedforward=self.attn_dim_feedforward,
                                                                  activation="relu"))
        self.transformer_encoder = nn.Sequential(*transformer_encoder)
        self.pre_attn = nn.Sequential(
            Rearrange('b c l -> l b c'),
            self.pos_encoder
        )
        self.post_attn = nn.Sequential(
            Rearrange('l b c -> b c l'),
            nn.BatchNorm1d(merge_dim),
            self.activation(),
            nn.Dropout(self.dropout),
            nn.Conv1d(merge_dim, 1, 1),
            self.activation(),
            nn.BatchNorm1d(1),
            Squeeze(dim=1),
        )
        
        # final output
        self.main_pred = nn.Sequential(
            nn.Linear(self.conv_tower_outlen, 1)
        )
        
    def forward(self, seq, chrom):
        seq = self.seq_stem(seq)
        seq = self.seq_conv_tower(seq)
        if not self.seqonly:
            chrom = self.chrom_stem(chrom)
            chrom = self.chrom_conv_tower(chrom)
            y_hat = torch.cat([seq, chrom], dim=1)
        else:
            y_hat = seq
        y_hat = self.pre_attn(y_hat)
        y_hat = self.transformer_encoder(y_hat)
        y_hat = self.post_attn(y_hat)
        y_pred = self.main_pred(y_hat)
        return y_pred
    
    def training_step(self, batch, batch_idx):
        # define train loop
        seq_r_s, chroms_r_s, y_r_s, label_r_s = batch
        
        pred_target_r_s = self(seq_r_s, chroms_r_s)

        # compute prediction and loss
        if self.classification:
            target_loss = F.binary_cross_entropy_with_logits(pred_target_r_s, label_r_s)
            accuracy = self.accuracy(F.sigmoid(pred_target_r_s), label_r_s)
            self.log('train_readcount_entropy_loss', target_loss)
            self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        else:
            target_loss = F.mse_loss(pred_target_r_s, y_r_s)
            self.log('train_readcount_MSE_loss', target_loss)
        return {'loss': target_loss}
    
    def validation_step(self, batch, batch_idx):
        # define validation loop
        seq, chroms, y, label = batch
        out = self(seq, chroms)
        if type(out) == tuple:
            y_target = out[0]
        else:
            y_target = out

        # compute prediction and loss
        if self.classification:
            val_loss = F.binary_cross_entropy_with_logits(y_target, label)
            accuracy = self.accuracy(F.sigmoid(y_target), label)
            self.log('val_loss', val_loss, sync_dist=True)
            self.log('val_accuracy', accuracy, sync_dist=True, on_epoch=True, prog_bar=True)
        else:    
            val_loss = F.mse_loss(y_target, y)
            self.log('val_loss', val_loss, sync_dist=True)
        return {'val_loss': val_loss}
        
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        # define test
        key, seq, chroms, y, label, domain = batch
        out = self(seq, chroms)
        if type(out) == tuple:
            y_target = out[0]
        else:
            y_target = out

        # compute prediction and loss
        if self.classification:
            test_loss = F.binary_cross_entropy_with_logits(y_target, label)
            self.test_step_outputs.append({'key': key, 'pred': y_target, 'true': label, 'domain': domain, 'dataloader_idx': torch.Tensor([dataloader_idx]).repeat(seq.shape[0])})
        else:
            test_loss = F.mse_loss(y_target, y)
            self.test_step_outputs.append({'key': key, 'pred': y_target, 'true': y, 'label': label, 'domain': domain, 'dataloader_idx': torch.Tensor([dataloader_idx]).repeat(seq.shape[0])})

        return test_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = CyclicLR(optimizer, max_lr=self.lr, base_lr=self.lr/10,
                             mode="exp_range", gamma=self.gamma,
                             cycle_momentum=False)
        
        return [optimizer], [{
                                "scheduler": scheduler,
                                "interval": "step",
                                "frequency": 1
                            }]

class ConvTowerDomain_v6_New_PostAttn(ConvTowerDomain_v6):
    """
    Avoid the over regularization in the post-attn layer in previous model
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.post_attn = nn.Sequential(
            Rearrange('l b c -> b c l'),
        )
        merge_dim = self.conv_tower_outdim if self.seqonly else self.conv_tower_outdim * 2
        
        # final output
        self.main_pred = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.conv_tower_outlen * merge_dim, 1),

        )

    def forward(self, seq, chrom):
        seq = self.seq_stem(seq)
        seq = self.seq_conv_tower(seq)
        if not self.seqonly:
            chrom = self.chrom_stem(chrom)
            chrom = self.chrom_conv_tower(chrom)
            y_hat = torch.cat([seq, chrom], dim=1)
        else:
            y_hat = seq
        y_hat = self.pre_attn(y_hat)
        y_hat = self.transformer_encoder(y_hat)
        y_hat = self.post_attn(y_hat)
        y_hat = torch.flatten(y_hat, start_dim=1)
        y_pred = self.main_pred(y_hat)
        return y_pred
    
class ConvTowerDomain_v6_GradientReversal_SplitSeqChrom_MultiDomain(ConvTowerDomain_v6):
    """
    Convolutional tower model v6 hooked by gradient reversal layer
    """
    def __init__(self, lambd_post_conv_tower_seq=0.0, lambd_post_conv_tower_chrom=0.0, num_cell_types=2, domain_predictor_type='attention', **kwargs):
        super().__init__(**kwargs)

        if domain_predictor_type=='attention':
            self.domain_predictor = self.construct_domain_attention_predictor
        elif domain_predictor_type=='fully_connected':
            self.domain_predictor = self.construct_domain_fully_connected_predictor
        else:
            raise Exception("domain_predictor_type has to be one of [attention, fully_connected]!")
        self.num_cell_types = num_cell_types
        self.lambd_post_conv_tower_seq = lambd_post_conv_tower_seq
        self.lambd_post_conv_tower_chrom = lambd_post_conv_tower_chrom

        self.post_conv_tower_seq_domain_pred = self.domain_predictor(in_feature=self.conv_tower_outdim, in_len=self.conv_tower_outlen, lambd=lambd_post_conv_tower_seq)
        self.post_conv_tower_chrom_domain_pred = self.domain_predictor(in_feature=self.conv_tower_outdim, in_len=self.conv_tower_outlen, lambd=lambd_post_conv_tower_chrom)

    def construct_domain_fully_connected_predictor(self, in_feature=1, in_len=500, lambd=0.):
        "Expect input in form [N, C, L]"
        # construct domain predictor
        domain_pred_list = []
        domain_pred_list.append(GradientReversal(lambd))
        if in_feature > 1:
            domain_pred_list.extend([
                nn.Conv1d(in_feature, 1, 1),
                self.activation(),
                nn.BatchNorm1d(1),
            ])
        domain_pred_list.append(Squeeze(dim=1))
        for i in range(3):
            domain_pred_list.append(
                DenseBlock(in_len, self.activation, dropout=self.dropout),
            )
        domain_pred_list.append(nn.Linear(in_len, self.num_cell_types))
        domain_predictor = nn.Sequential(*domain_pred_list)
        
        return domain_predictor
    

    def construct_domain_attention_predictor(self, in_feature= 1, in_len=500, lambd=0.,
                                             attn_num_layers = 2, attn_num_heads=8,
                                             attn_dim_feedforward = 512): 
        "Expect input in the shape of [N, C, L]"
        # self attention
        pos_encoder = PositionalEncoding(in_feature, self.dropout, in_len)
        transformer_encoder = []
        for i in range(attn_num_layers):
            transformer_encoder.append(nn.TransformerEncoderLayer(d_model=in_feature,
                                                                  nhead=attn_num_heads,
                                                                  dim_feedforward=attn_dim_feedforward,
                                                                  activation="relu"))
        transformer_encoder = nn.Sequential(*transformer_encoder)
        pre_attn = nn.Sequential(
            GradientReversal(lambd),
            Rearrange('b c l -> l b c'),
            pos_encoder
        )
        post_attn = nn.Sequential(
            Rearrange('l b c -> b c l'),
            nn.BatchNorm1d(in_feature),
            self.activation(),
            nn.Dropout(self.dropout),
            nn.Conv1d(in_feature, 1, 1),
            self.activation(),
            nn.BatchNorm1d(1),
            Squeeze(dim=1),
        )
        
        # final output
        main_pred = nn.Sequential(
            nn.Linear(in_len, self.num_cell_types)
        )

        domain_attn_predictor = nn.Sequential(pre_attn,
                                              transformer_encoder,
                                              post_attn,
                                              main_pred)
        return domain_attn_predictor
    
    def construct_domain_predictor(self, in_feature=1, in_len=500, lambd=0.):
        "Expect input in form [N, C, L]"
        # construct domain predictor
        domain_pred_list = []
        domain_pred_list.append(GradientReversal(lambd))
        if in_feature > 1:
            domain_pred_list.extend([
                nn.Conv1d(in_feature, 1, 1),
                self.activation(),
                nn.BatchNorm1d(1),
            ])
        domain_pred_list.append(Squeeze(dim=1))
        for i in range(3):
            domain_pred_list.append(
                DenseBlock(in_len, self.activation, dropout=self.dropout),
            )
        domain_pred_list.append(nn.Linear(in_len, self.num_cell_types))
        domain_predictor = nn.Sequential(*domain_pred_list)
        
        return domain_predictor
    
    def forward(self, seq, chrom):
        seq = self.seq_stem(seq)
        seq = self.seq_conv_tower(seq); y_domain_seq = self.post_conv_tower_seq_domain_pred(seq)
        if not self.seqonly:
            chrom = self.chrom_stem(chrom)
            chrom = self.chrom_conv_tower(chrom); y_domain_chrom = self.post_conv_tower_chrom_domain_pred(chrom)
            y_hat = torch.cat([seq, chrom], dim=1)
        else:
            y_hat = seq
        y_hat = self.pre_attn(y_hat)
        y_hat = self.transformer_encoder(y_hat)
        y_hat = self.post_attn(y_hat)
        y_pred = self.main_pred(y_hat)
        if not self.seqonly:
            return y_pred, y_domain_seq, y_domain_chrom
        else:
            return y_pred, y_domain_seq
    
    def training_step(self, batch, batch_idx):
        """
        [seq, chrom, y, label] marks the type of input;
        [r_s, r_t, d] marks the prediction task (readcount task/domain task) and domain
        [seq, chrom] marks which domain task will use this data
        """
        # define train loop
        seq_r_s_b, chroms_r_s_b, y_r_s_b, label_r_s_b = batch['readcount_source_bound']
        seq_r_s_ub, chroms_r_s_ub, y_r_s_ub, label_r_s_ub = batch['readcount_source_unbound']
        seq_d_seq, chroms_d_seq, y_d_seq, label_d_seq = batch['domain_seq']
        seq_d_chrom, chroms_d_chrom, y_d_chrom, label_d_chrom = batch['domain_chrom']
       
        # merge inputs
        seq_merge = torch.cat((seq_r_s_b, seq_r_s_ub, seq_d_seq, seq_d_chrom))
        chroms_merge = torch.cat((chroms_r_s_b, chroms_r_s_ub, chroms_d_seq, chroms_d_chrom))
        
        pred_target_merge, pred_domain_conv_tower_seq_merge, pred_domain_conv_tower_chrom_merge = self(seq_merge, chroms_merge)
        r_s_size = seq_r_s_b.shape[0] + seq_r_s_ub.shape[0]
        domain_seq_size = seq_d_seq.shape[0]
        pred_target_r_s = pred_target_merge[:r_s_size]
        pred_domain_conv_tower_seq_merge = pred_domain_conv_tower_seq_merge[r_s_size:r_s_size+domain_seq_size]
        pred_domain_conv_tower_chrom_merge = pred_domain_conv_tower_chrom_merge[r_s_size+domain_seq_size:]

        # compute prediction and loss
        if self.classification:
            target_loss = F.binary_cross_entropy_with_logits(pred_target_r_s, torch.cat((label_r_s_b, label_r_s_ub)))
            accuracy = self.accuracy(F.sigmoid(pred_target_r_s), torch.cat((label_r_s_b, label_r_s_ub)))
            self.log('train_readcount_entropy_loss', target_loss)
            self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True)
        else:
            target_loss = F.mse_loss(pred_target_r_s, torch.cat((y_r_s_b, y_r_s_ub)))
            self.log('train_readcount_MSE_loss', target_loss)
        domain_conv_tower_seq_loss = F.cross_entropy(pred_domain_conv_tower_seq_merge, label_d_seq.squeeze(dim=1)) if self.lambd_post_conv_tower_seq > 0  else 0
        domain_conv_tower_chrom_loss = F.cross_entropy(pred_domain_conv_tower_chrom_merge, label_d_chrom.squeeze(dim=1)) if self.lambd_post_conv_tower_chrom > 0 else 0
        self.log('train_domain_conv_tower_seq_entropy_loss', domain_conv_tower_seq_loss)
        self.log('train_domain_conv_tower_chrom_entropy_loss', domain_conv_tower_chrom_loss)
        return {'loss': target_loss + domain_conv_tower_seq_loss + domain_conv_tower_chrom_loss}

class ConvTowerDomain_v6_ADDA(ConvTowerDomain_v6):
    """
    ADDA on v6 model, only domain adapt chromatin subnetwork
    """
    def __init__(self, ckpt_file, **kwargs):
        super().__init__(**kwargs)

        self.ckpt_file = ckpt_file
        self.backbone = ConvTowerDomain_v6.load_from_checkpoint(self.ckpt_file)

        self.discriminator = construct_domain_attention_discriminator(num_class=1, # only need 1 prediction when binary
                                                                  in_feature=self.backbone.conv_tower_outdim,
                                                                  in_len=self.backbone.conv_tower_outlen)
        self.classifier = nn.Sequential(self.backbone.pre_attn,
                                        self.backbone.transformer_encoder,
                                        self.backbone.post_attn,
                                        self.backbone.main_pred)
        
        self.seq_feature_extractor = nn.Sequential(self.backbone.seq_stem,
                                                   self.backbone.seq_conv_tower)
        self.chrom_feature_extractor = nn.Sequential(self.backbone.chrom_stem,
                                                   self.backbone.chrom_conv_tower)
        self.chrom_feature_extractor_tgt = deepcopy(self.chrom_feature_extractor)

        # freeze non-trainable parts
        self.seq_feature_extractor.eval()
        self.chrom_feature_extractor.eval()
        self.classifier.eval()

        self.automatic_optimization = False
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, seq, chrom):
        seq = self.seq_feature_extractor(seq)
        chrom = self.chrom_feature_extractor_tgt(chrom)
        y_hat = torch.cat([seq, chrom], dim=1)
        y_pred = self.classifier(y_hat)

        return y_pred

    def training_step(self, batch, batch_idx):
        # get optimizers
        tgt_encoder_opt, d_opt = self.optimizers()
        # get inputs
        seq_d_s, chroms_d_s, y_d_s, label_d_s = batch['domain_source']
        seq_d_t, chroms_d_t, y_d_t, label_d_t = batch['domain_target']
        # create labels
        batch_size = seq_d_s.shape[0]
        source_label = torch.ones((batch_size, 1), device=self.device)
        target_label = torch.zeros((batch_size, 1), device=self.device)
        # define predict function
        def predict_d_s(seq, chrom):
            chrom = self.chrom_feature_extractor(chrom)
            y_hat = self.discriminator(chrom)
            return y_hat
        def predict_d_t(seq, chrom):
            chrom = self.chrom_feature_extractor_tgt(chrom)
            y_hat = self.discriminator(chrom)
            return y_hat
        ########################
        # Optimize Discriminator
        ########################
        # get predictions
        y_hat_s = predict_d_s(seq_d_s, chroms_d_s)
        y_hat_t = predict_d_t(seq_d_t, chroms_d_t)
        loss_s = self.criterion(y_hat_s, source_label)
        loss_t = self.criterion(y_hat_t, target_label)
        loss = loss_s + loss_t

        d_opt.zero_grad()
        self.manual_backward(loss)
        d_opt.step()

        ########################
        # Optimize Encoder
        ########################
        y_hat_t = predict_d_t(seq_d_t, chroms_d_t)
        loss = self.criterion(y_hat_t, source_label)

        tgt_encoder_opt.zero_grad()
        self.manual_backward(loss)
        tgt_encoder_opt.step()

        # Step learning rate scheduler 
        if self.trainer.is_last_batch:
            scheduler_tgt_encoder, scheduler_d = self.lr_schedulers()
            scheduler_tgt_encoder.step()
            scheduler_d.step()

        self.log_dict({"loss_source": loss_s, "loss_target": loss_t})

        return loss
    
    def configure_optimizers(self):

        tgt_encoder_opt = optim.AdamW(self.chrom_feature_extractor_tgt.parameters(), lr=1e-5)
        scheduler_tgt_encoder = SequentialLR(tgt_encoder_opt, 
                                   schedulers=[LambdaLR(tgt_encoder_opt, lr_lambda=lambda epoch: 0),
                                               OneCycleLR(tgt_encoder_opt, max_lr=1e-4, total_steps=40),
                                               LambdaLR(tgt_encoder_opt, lr_lambda=lambda epoch: 1)],
                                   milestones=[10, 40])

        d_opt = optim.AdamW(self.discriminator.parameters(), lr=1e-5)
        scheduler_d = StepLR(d_opt, step_size=5, gamma=0.5)

        return [tgt_encoder_opt, d_opt], [scheduler_tgt_encoder, scheduler_d]
        
class ConvTowerDomain_v6_ADDA_ACC(ConvTowerDomain_v6):
    """
    ADDA on v6 model, only domain adapt chromatin subnetwork, separate accessible and inaccessible data
    """
    def __init__(self, ckpt_file, **kwargs):
        super().__init__(**kwargs)

        self.ckpt_file = ckpt_file
        self.backbone = ConvTowerDomain_v6.load_from_checkpoint(self.ckpt_file)

        self.discriminator = construct_domain_attention_discriminator(num_class=4, # only need 1 prediction when binary
                                                                  in_feature=self.backbone.conv_tower_outdim,
                                                                  in_len=self.backbone.conv_tower_outlen)
        self.classifier = nn.Sequential(self.backbone.pre_attn,
                                        self.backbone.transformer_encoder,
                                        self.backbone.post_attn,
                                        self.backbone.main_pred)
        
        self.seq_feature_extractor = nn.Sequential(self.backbone.seq_stem,
                                                   self.backbone.seq_conv_tower)
        self.chrom_feature_extractor = nn.Sequential(self.backbone.chrom_stem,
                                                   self.backbone.chrom_conv_tower)
        self.chrom_feature_extractor_tgt = deepcopy(self.chrom_feature_extractor)

        # freeze non-trainable parts
        self.seq_feature_extractor.eval()
        self.chrom_feature_extractor.eval()
        self.classifier.eval()

        self.automatic_optimization = False
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, seq, chrom):
        seq = self.seq_feature_extractor(seq)
        chrom = self.chrom_feature_extractor_tgt(chrom)
        y_hat = torch.cat([seq, chrom], dim=1)
        y_pred = self.classifier(y_hat)

        return y_pred

    def training_step(self, batch, batch_idx):
        # get optimizers
        tgt_encoder_opt, d_opt = self.optimizers()
        # get inputs
        seq_d_s_acc, chroms_d_s_acc, y_d_s_acc, label_d_s_acc = batch['domain_source_acc']
        seq_d_s_inacc, chroms_d_s_inacc, y_d_s_inacc, label_d_s_inacc = batch['domain_source_inacc']
        seq_d_t_acc, chroms_d_t_acc, y_d_t_acc, label_d_t_acc = batch['domain_target_acc']
        seq_d_t_inacc, chroms_d_t_inacc, y_d_t_inacc, label_d_t_inacc = batch['domain_target_inacc']
        # create labels
        batch_size = seq_d_s_acc.shape[0]
        source_acc_label = torch.full((batch_size,), 0, device=self.device)
        source_inacc_label = torch.full((batch_size,), 1, device=self.device)
        target_acc_label = torch.full((batch_size,), 2, device=self.device)
        target_inacc_label = torch.full((batch_size,), 3, device=self.device)
        # define predict function
        def predict_d_s(seq, chrom):
            chrom = self.chrom_feature_extractor(chrom)
            y_hat = self.discriminator(chrom)
            return y_hat
        def predict_d_t(seq, chrom):
            chrom = self.chrom_feature_extractor_tgt(chrom)
            y_hat = self.discriminator(chrom)
            return y_hat
        ########################
        # Optimize Discriminator
        ########################
        # get predictions
        y_hat_s_acc = predict_d_s(seq_d_s_acc, chroms_d_s_acc)
        y_hat_s_inacc = predict_d_s(seq_d_s_inacc, chroms_d_s_inacc)
        y_hat_t_acc = predict_d_t(seq_d_t_acc, chroms_d_t_acc)
        y_hat_t_inacc = predict_d_t(seq_d_t_inacc, chroms_d_t_inacc)
        loss_s_acc = self.criterion(y_hat_s_acc, source_acc_label)
        loss_s_inacc = self.criterion(y_hat_s_inacc, source_inacc_label)
        loss_t_acc = self.criterion(y_hat_t_acc, target_acc_label)
        loss_t_inacc = self.criterion(y_hat_t_inacc, target_inacc_label)
        loss = loss_s_acc + loss_s_inacc + loss_t_acc + loss_t_inacc

        d_opt.zero_grad()
        self.manual_backward(loss)
        d_opt.step()

        ########################
        # Optimize Encoder
        ########################
        y_hat_t_acc = predict_d_t(seq_d_t_acc, chroms_d_t_acc)
        y_hat_t_inacc = predict_d_t(seq_d_t_inacc, chroms_d_t_inacc)
        loss_acc = self.criterion(y_hat_t_acc, source_acc_label)
        loss_inacc = self.criterion(y_hat_t_inacc, source_inacc_label)

        tgt_encoder_opt.zero_grad()
        self.manual_backward(loss_acc + loss_inacc)
        tgt_encoder_opt.step()

        # Step learning rate scheduler 
        if self.trainer.is_last_batch:
            scheduler_tgt_encoder, scheduler_d = self.lr_schedulers()
            scheduler_tgt_encoder.step()
            scheduler_d.step()

        self.log_dict({"loss_source_acc": loss_s_acc, "loss_target_acc": loss_t_acc,
                       "loss_source_inacc": loss_s_inacc, "loss_target_inacc": loss_t_inacc})

        return loss
    
    def configure_optimizers(self):

        tgt_encoder_opt = optim.AdamW(self.chrom_feature_extractor_tgt.parameters(), lr=1e-5)
        scheduler_tgt_encoder = SequentialLR(tgt_encoder_opt, 
                                   schedulers=[LambdaLR(tgt_encoder_opt, lr_lambda=lambda epoch: 0),
                                               OneCycleLR(tgt_encoder_opt, max_lr=1e-4, total_steps=40),
                                               LambdaLR(tgt_encoder_opt, lr_lambda=lambda epoch: 1)],
                                   milestones=[10, 40])

        d_opt = optim.AdamW(self.discriminator.parameters(), lr=1e-5)
        scheduler_d = StepLR(d_opt, step_size=5, gamma=0.5)

        return [tgt_encoder_opt, d_opt], [scheduler_tgt_encoder, scheduler_d]

class ConvTowerDomain_v6_ADDA_SeqChrom(ConvTowerDomain_v6):
    """
    ADDA on v6 model, can do both sequence and chromatin adda
    """
    def __init__(self, ckpt_file, seq_adda=False, chrom_adda=True, **kwargs):
        super().__init__(**kwargs)

        self.ckpt_file = ckpt_file
        self.seq_adda = seq_adda
        self.chrom_adda = chrom_adda
        self.backbone = ConvTowerDomain_v6.load_from_checkpoint(self.ckpt_file)
            
        self.discriminator_seq = construct_domain_attention_discriminator(num_class=1, # only need 1 prediction when binary
                                                                  in_feature=self.backbone.conv_tower_outdim,
                                                                  in_len=self.backbone.conv_tower_outlen)
        self.discriminator_chrom = deepcopy(self.discriminator_seq)
        self.classifier = nn.Sequential(self.backbone.pre_attn,
                                        self.backbone.transformer_encoder,
                                        self.backbone.post_attn,
                                        self.backbone.main_pred)
        
        self.seq_feature_extractor = nn.Sequential(self.backbone.seq_stem,
                                                   self.backbone.seq_conv_tower)
        self.chrom_feature_extractor = nn.Sequential(self.backbone.chrom_stem,
                                                   self.backbone.chrom_conv_tower)
        self.seq_feature_extractor_tgt = deepcopy(self.seq_feature_extractor)
        self.chrom_feature_extractor_tgt = deepcopy(self.chrom_feature_extractor)

        # freeze non-trainable parts
        if not seq_adda:
            self.seq_feature_extractor_tgt.eval()
        if not chrom_adda:
            self.chrom_feature_extractor_tgt.eval()
        self.seq_feature_extractor.eval()
        self.chrom_feature_extractor.eval()
        self.classifier.eval()

        self.automatic_optimization = False
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, seq, chrom):
        seq = self.seq_feature_extractor_tgt(seq)
        chrom = self.chrom_feature_extractor_tgt(chrom)
        y_hat = torch.cat([seq, chrom], dim=1)
        y_pred = self.classifier(y_hat)

        return y_pred

    def training_step(self, batch, batch_idx):
        # get optimizers
        tgt_encoder_opt_seq, tgt_encoder_opt_chrom, d_opt_seq, d_opt_chrom = self.optimizers()
        # get inputs
        seq_d_s, chroms_d_s, y_d_s, label_d_s = batch['domain_source']
        seq_d_t, chroms_d_t, y_d_t, label_d_t = batch['domain_target']
        # create labels
        batch_size = seq_d_s.shape[0]
        source_label = torch.ones((batch_size, 1), device=self.device)
        target_label = torch.zeros((batch_size, 1), device=self.device)
        # define predict function
        def predict_d_s(seq, chrom):
            seq = self.seq_feature_extractor(seq)
            chrom = self.chrom_feature_extractor(chrom)
            y_hat_seq = self.discriminator_seq(seq)
            y_hat_chrom = self.discriminator_chrom(chrom)
            return y_hat_seq, y_hat_chrom
        def predict_d_t(seq, chrom):
            seq = self.seq_feature_extractor_tgt(seq)
            chrom = self.chrom_feature_extractor_tgt(chrom)
            y_hat_seq = self.discriminator_seq(seq)
            y_hat_chrom = self.discriminator_chrom(chrom)
            return y_hat_seq, y_hat_chrom
        ########################
        # Optimize Discriminator
        ########################
        # get predictions
        y_hat_s_seq, y_hat_s_chrom = predict_d_s(seq_d_s, chroms_d_s)
        y_hat_t_seq, y_hat_t_chrom = predict_d_t(seq_d_t, chroms_d_t)
        loss_s_seq = self.criterion(y_hat_s_seq, source_label)
        loss_s_chrom = self.criterion(y_hat_s_chrom, source_label)
        loss_t_seq = self.criterion(y_hat_t_seq, target_label)
        loss_t_chrom = self.criterion(y_hat_t_chrom, target_label)
        loss = loss_s_seq + loss_s_chrom + loss_t_seq + loss_t_chrom

        d_opt_seq.zero_grad()
        d_opt_chrom.zero_grad()
        self.manual_backward(loss)
        d_opt_seq.step()
        d_opt_chrom.step()

        ########################
        # Optimize Encoder
        ########################
        y_hat_t_seq, y_hat_t_chrom = predict_d_t(seq_d_t, chroms_d_t)
        loss = torch.tensor(0, device=self.device, dtype=loss.dtype)
        if self.seq_adda:
            loss += self.criterion(y_hat_t_seq, source_label)
        if self.chrom_adda:
            loss += self.criterion(y_hat_t_chrom, source_label)

        tgt_encoder_opt_seq.zero_grad()
        tgt_encoder_opt_chrom.zero_grad()
        self.manual_backward(loss)
        if self.seq_adda:
            tgt_encoder_opt_seq.step()
        if self.chrom_adda:
            tgt_encoder_opt_chrom.step()

        # Step learning rate scheduler 
        if self.trainer.is_last_batch:
            scheduler_tgt_encoder_seq, scheduler_tgt_encoder_chrom, scheduler_d_seq, scheduler_d_chrom = self.lr_schedulers()
            scheduler_tgt_encoder_seq.step()
            scheduler_tgt_encoder_chrom.step()
            scheduler_d_seq.step()
            scheduler_d_chrom.step()

        self.log_dict({"loss_source_seq": loss_s_seq, "loss_source_chrom": loss_s_chrom,
                       "loss_target_seq": loss_t_seq, "loss_target_chrom": loss_t_chrom})

        return loss
    
    def configure_optimizers(self):

        tgt_encoder_opt_seq = optim.AdamW(self.seq_feature_extractor_tgt.parameters(), lr=1e-5)
        scheduler_tgt_encoder_seq = SequentialLR(tgt_encoder_opt_seq, 
                                   schedulers=[LambdaLR(tgt_encoder_opt_seq, lr_lambda=lambda epoch: 0),
                                               OneCycleLR(tgt_encoder_opt_seq, max_lr=1e-4, total_steps=40),
                                               LambdaLR(tgt_encoder_opt_seq, lr_lambda=lambda epoch: 1)],
                                   milestones=[10, 40])

        tgt_encoder_opt_chrom = optim.AdamW(self.chrom_feature_extractor_tgt.parameters(), lr=1e-5)
        scheduler_tgt_encoder_chrom = SequentialLR(tgt_encoder_opt_chrom, 
                                   schedulers=[LambdaLR(tgt_encoder_opt_chrom, lr_lambda=lambda epoch: 0),
                                               OneCycleLR(tgt_encoder_opt_chrom, max_lr=1e-4, total_steps=40),
                                               LambdaLR(tgt_encoder_opt_chrom, lr_lambda=lambda epoch: 1)],
                                   milestones=[10, 40])

        d_opt_seq = optim.AdamW(self.discriminator_seq.parameters(), lr=1e-5)
        d_opt_chrom = optim.AdamW(self.discriminator_chrom.parameters(), lr=1e-5)
        scheduler_d_seq = StepLR(d_opt_seq, step_size=5, gamma=0.5)
        scheduler_d_chrom = StepLR(d_opt_chrom, step_size=5, gamma=0.5)

        return [tgt_encoder_opt_seq, tgt_encoder_opt_chrom, d_opt_seq, d_opt_chrom], [scheduler_tgt_encoder_seq, scheduler_tgt_encoder_chrom, scheduler_d_seq, scheduler_d_chrom]

def construct_domain_attention_discriminator(dropout=0.5, in_feature=1, in_len=500,
                                         attn_num_layers=2, attn_num_heads=8,
                                         attn_dim_feedforward = 512, num_class=2): 
    "Expect input in the shape of [N, C, L]"
    # self attention
    pos_encoder = PositionalEncoding(in_feature, dropout, in_len)
    transformer_encoder = []
    for i in range(attn_num_layers):
        transformer_encoder.append(nn.TransformerEncoderLayer(d_model=in_feature,
                                                              nhead=attn_num_heads,
                                                              dim_feedforward=attn_dim_feedforward,
                                                              activation="relu"))
    transformer_encoder = nn.Sequential(*transformer_encoder)
    pre_attn = nn.Sequential(
        Rearrange('b c l -> l b c'),
        pos_encoder
    )
    post_attn = nn.Sequential(
        Rearrange('l b c -> b c l'),
        nn.BatchNorm1d(in_feature),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Conv1d(in_feature, 1, 1),
        nn.GELU(),
        nn.BatchNorm1d(1),
        Squeeze(dim=1),
    )
    
    # final output
    main_pred = nn.Sequential(
        nn.Linear(in_len, num_class)
    )

    domain_attn_predictor = nn.Sequential(pre_attn,
                                          transformer_encoder,
                                          post_attn,
                                          main_pred)
    return domain_attn_predictor

class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_optimizer_args(torch.optim.AdamW)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)

def cli_main():
    from datetime import datetime

    run_name = os.environ.get("RUN_NAME")
    if not run_name:
        run_name = datetime.now().strftime("FOXA1_%Y%m%d_%H%M%S")
        os.environ["RUN_NAME"] = run_name
    if "--trainer.logger.init_args.name" not in sys.argv:
        sys.argv.extend(["--trainer.logger.init_args.name", run_name])

    ckpt_dir = os.path.join("checkpoints", run_name)
    print(f"RUN_NAME={run_name}")
    print(f"Checkpoint directory: {ckpt_dir}")
    os.makedirs(ckpt_dir, exist_ok=True)
    cli = LightningCLI(seed_everything_default=32,
                       save_config_kwargs={"overwrite": True},
                         trainer_defaults={
                            "callbacks": [
                                # 每个 epoch 训练结束后覆盖最新权重，不依赖 validation
                                ModelCheckpoint(
                                    dirpath=ckpt_dir,
                                    filename="last-{epoch:02d}",
                                    save_top_k=0,
                                    save_last=True,
                                    every_n_epochs=1,
                                    save_on_train_epoch_end=True,
                                    verbose=True,
                                ),
                                # 验证后只保留 val_loss 最低的一份
                                ModelCheckpoint(
                                    dirpath=ckpt_dir,
                                    filename="best-{epoch:02d}-{val_loss:.6f}",
                                    monitor="val_loss",
                                    mode="min",
                                    save_top_k=1,
                                    save_last=False,
                                    every_n_epochs=1,
                                    save_on_train_epoch_end=False,
                                    verbose=True,
                                ),
                                ModelSummary(max_depth=-1),
                                LearningRateMonitor(logging_interval='step')]
                        })
    
if __name__ == "__main__":
    cli_main()
