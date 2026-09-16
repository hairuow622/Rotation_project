#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seqchromloader as scl
import torch
import webdataset as wds

from braceexpand import braceexpand
from functools import partial
from numpy import random
from pprint import pp
from pytorch_lightning.utilities.combined_loader import CombinedLoader
from yaml import safe_load

from torch.utils.data import DataLoader, default_collate, ChainDataset

scl.mute_warning()

class DataConfig():
    def __init__(self, config, parent_dir="./"):
        self.config = safe_load(open(config))
        self.parent_dir = parent_dir

    def get_wds_files(self, data_subset, data_type):
        try:
            return list(braceexpand(self.config['webdataset'][data_subset][data_type]))
        except KeyError:
            pp(self.config['webdataset'])
            raise KeyError(f"Couldn't find webdataset files corresponding to {data_subset}/{data_type}")

    def get_wds_files_by_subset(self, data_subset):
        fs = []
        for data_type in self.config["webdataset"][data_subset].keys():
            fs.extend(self.get_wds_files(data_subset, data_type))
        return fs

    def get_bed_file(self, data_subset, data_type):
        return self.config['bed'][data_subset][data_type]

    def get_datapipe_by_subset(self, data_subset, resample=False, transforms=None, keep_key=False):
        dp = self.build_wds_pipeline(self.get_wds_files_by_subset(data_subset),
                                     resample=resample, transforms=transforms, keep_key=keep_key)

        return dp

    def get_datapipe(self, data_subset, data_type, resample=False, transforms=None, keep_key=False):
        dp = self.build_wds_pipeline(self.get_wds_files(data_subset, data_type),
                                     resample=resample, transforms=transforms, keep_key=keep_key)
        return dp

    def build_wds_pipeline(self, wds_files, resample=False, transforms=None, keep_key=False):
        if wds_files is None:
            UserWarning("wds_files is None, return None without building wds pipeline, make sure this is what you expect!")
            return None

        # concatenate the parent directory
        wds_files = ([os.path.join(self.parent_dir, w) for w in wds_files] if isinstance(wds_files, list)
                                                                    else os.path.join(self.parent_dir, wds_files))
    
        pipeline = []
        if resample:
              pipeline.extend([
                  wds.ResampledShards(wds_files),
                  wds.tarfile_to_samples(),                                                                                                                                                                                                                     wds.shuffle(50000)   # I realized seed is not necessary here because pytorch lightning takes care of seeding in worker https://github.com/Lightning-AI/lightning/blob/984f49f7195ddc67e961c7c498ee6e19fc0cecb5/src/lightning/fab  ric/utilities/seed.py#L81
              ])
        else:
              pipeline.extend([
                  wds.SimpleShardList(wds_files),
                  wds.split_by_node,
                  wds.split_by_worker,
                  wds.shuffle(200),  # shuffle must be after split_by_* to ensure non-overlapping shards across workers!
                  wds.tarfile_to_samples(),
                  wds.shuffle(50000)   # It's necessary to have a bigger sample pool to ensure the samples from different files are mixed in batches
              ])
        pipeline.extend([
              wds.decode(),
              wds.rename(seq="seq.npy",
                         chrom="chrom.npy",
                         target="target.npy",
                         label="label.npy"),
          ])
        if transforms:
              pipeline.append(wds.map_dict(**transforms))
    
        if keep_key:
            def split_key(i):
                return i.split(',')
            pipeline.append(wds.map_dict(__key__= split_key))
            pipeline.append(wds.to_tuple('__key__', 'seq', 'chrom', 'target', 'label'))
        else:
            pipeline.append(wds.to_tuple('seq', 'chrom', 'target', 'label'))
    
        pipeline.append(wds.unbatched())
        pipeline.append(wds.shuffle(50000))
    
        return wds.DataPipeline(pipeline)


class MergedLoader:
    def __init__(self, loader1, loader2, length):
        self.loader1 = loader1
        self.loader2 = loader2
        self.loader1_iter = iter(loader1)
        self.loader2_iter = iter(loader2)
        self.batch_size = loader1.batch_size + loader2.batch_size
        self.length = length

    def __iter__(self):
        count = 0
        while True:
            if count >= self.length: raise StopIteration
            count += 1
            try:
                batch1 = next(self.loader1_iter)
            except StopIteration:
                self.loader1_iter = iter(self.loader1)
                batch1 = next(self.loader1_iter)
                
            try:
                batch2 = next(self.loader2_iter)
            except StopIteration:
                self.loader2_iter = iter(self.loader2)
                batch2 = next(self.loader2_iter)
                
            # Combine batches from loader1 and loader2 (adjust as needed)
            batch_comb = tuple(torch.cat(batch1[idx], batch2[idx]) for idx in range(len(batch1)))
                
            yield batch_comb

    def __len__(self):
        return self.length

class ChainedLoader:
    def __init__(self, loaders:list):
        """
        Iterate through each loader, with fixed total length
        """
        self.loaders = loaders

    def __iter__(self):
        for loader in self.loaders:
            for b in iter(loader):
                yield b

class ChainedLoaderSample:
    def __init__(self, loaders:list, length):
        """
        Iterate through each loader, with fixed total length
        """
        self.loaders = loaders
        self.loaders_iter = [iter(l) for l in loaders]
        self.batch_size = loaders[0].batch_size
        self.length = length

    def __iter__(self):
        count = 0
        while True:
            if count >= self.length: raise StopIteration
            count += 1

            loader_choice = random.choice(range(len(self.loaders)))
            
            try:
                yield next(self.loaders_iter[loader_choice])
            except StopIteration:
                self.loaders_iter[loader_choice] = iter(self.loaders[loader_choice])
                yield next(self.loaders_iter[loader_choice])

def collate_add_domain(batch, domain=0):
    batch = default_collate(batch)
    domain = torch.full_like(batch[-1], domain, dtype=torch.float32)  # assume the last element is label
    
    return *batch, domain.float()

collate_add_domain_source = partial(collate_add_domain, domain=0)
collate_add_domain_target = partial(collate_add_domain, domain=1)

class SingleDataModuleWds(pl.LightningDataModule):
    def __init__(self, config_file, parent_dir="./", num_workers=4, batch_size=512, steps_per_epoch=1000):
        super().__init__()

        self.dataconfig = DataConfig(config_file, parent_dir=parent_dir)

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        try:
            device_id = self.trainer.device_ids[self.trainer.local_rank]
        
            world_size = self.trainer.world_size
            print(f"device id {device_id}, local rank {self.trainer.local_rank}, global rank {self.trainer.global_rank} in world {world_size}")
        except AttributeError:
            print(f"Error when trying to fetch device and rank info")
            print(f"Assume dataset is being setup without a trainer, set device id as 0, global rank as 0, world size as 1")
            device_id = 0
            world_size = 1
        self.batch_size_per_rank = int(self.batch_size/world_size)

        if stage == 'fit':
            self.train_r_b = self.dataconfig.get_datapipe_by_subset('single_train_readcount_bound', resample=True)
            self.train_r_ub = self.dataconfig.get_datapipe_by_subset('single_train_readcount_unbound', resample=True)
            self.train_r = wds.RandomMix([self.train_r_b, self.train_r_ub]) # mix bound and unbound training set with equal probabilities
            self.val_r = self.dataconfig.get_datapipe_by_subset('single_val_readcount', resample=False)

        if stage == 'test':
            self.test_readcount = self.dataconfig.get_datapipe_by_subset('single_test_readcount', resample=False, keep_key=True)
            self.test_rand = self.dataconfig.get_datapipe_by_subset('single_test_random', resample=False, keep_key=True)

    def train_dataloader(self):
        prefetch_factor = 32
        train_r_loader = wds.WebLoader(self.train_r, num_workers=self.num_workers,
                                       batch_size=self.batch_size_per_rank,
                                       pin_memory=True, prefetch_factor=prefetch_factor).with_epoch(self.steps_per_epoch)

        return train_r_loader

    def val_dataloader(self):
        return wds.WebLoader(self.val_r , num_workers=self.num_workers, 
                             batch_size=self.batch_size_per_rank, pin_memory=True)

    def test_dataloader(self):
        return [wds.WebLoader(self.test_readcount, num_workers=self.num_workers, 
                             batch_size=self.batch_size_per_rank, 
                             collate_fn=collate_add_domain_source, pin_memory=True),
                wds.WebLoader(self.test_rand, num_workers=self.num_workers, 
                             batch_size=self.batch_size_per_rank, 
                             collate_fn=collate_add_domain_source, pin_memory=True)]

class MultiDataModuleWds(pl.LightningDataModule):
    def __init__(self, configs:list, parent_dir="./", num_workers=4, batch_size=512, steps_per_epoch=1000):
        super().__init__()

        self.dataconfigs = [DataConfig(c, parent_dir=parent_dir) for c in configs]

        self.num_workers = num_workers
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        try:
            device_id = self.trainer.device_ids[self.trainer.local_rank]
        
            world_size = self.trainer.world_size
            print(f"device id {device_id}, local rank {self.trainer.local_rank}, global rank {self.trainer.global_rank} in world {world_size}")
        except AttributeError:
            print(f"Error when trying to fetch device and rank info")
            print(f"Assume dataset is being setup without a trainer, set device id as 0, global rank as 0, world size as 1")
            device_id = 0
            world_size = 1
        self.batch_size_per_rank = int(self.batch_size/world_size)
        
        if stage=='fit':
            self.train_r_b = [dc.get_datapipe_by_subset('single_train_readcount_bound', resample=True) for dc in self.dataconfigs]
            self.train_r_ub = [dc.get_datapipe_by_subset('single_train_readcount_unbound', resample=True) for dc in self.dataconfigs]
            self.train_r = wds.RandomMix([*self.train_r_b, *self.train_r_ub]) # mix bound and unbound training set with equal probabilities
            self.val_r = ChainDataset([dc.get_datapipe_by_subset('single_val_readcount', resample=False) for dc in self.dataconfigs])
        elif stage=='test':
            self.test_readcount = ChainDataset([dc.get_datapipe_by_subset('single_test_readcount', resample=False, keep_key=True) for dc in self.dataconfigs])
            self.test_rand = ChainDataset([dc.get_datapipe_by_subset('single_test_random', resample=False, keep_key=True) for dc in self.dataconfigs])

    def train_dataloader(self):
        prefetch_factor = 32
        train_r_loader = wds.WebLoader(self.train_r, num_workers=self.num_workers,
                                       batch_size=self.batch_size_per_rank,
                                       pin_memory=True, prefetch_factor=prefetch_factor).with_epoch(self.steps_per_epoch)

        return train_r_loader

    def val_dataloader(self):
        return wds.WebLoader(self.val_r , num_workers=self.num_workers, 
                             batch_size=self.batch_size_per_rank, pin_memory=True)

    def test_dataloader(self):
        return [wds.WebLoader(self.test_readcount, num_workers=self.num_workers, 
                             batch_size=self.batch_size_per_rank, 
                             collate_fn=collate_add_domain_source, pin_memory=True),
                wds.WebLoader(self.test_rand, num_workers=self.num_workers, 
                             batch_size=self.batch_size_per_rank, 
                             collate_fn=collate_add_domain_source, pin_memory=True)]


class DomainDataModuleWds(pl.LightningDataModule):
    def __init__(self,
                 source_configs:list,
                 target_config:str,
                 parent_dir="./",
                 num_workers=1, 
                 batch_size=512, steps_per_epoch=1000):
        super().__init__()
        
        self.source_dataconfigs = [DataConfig(s, parent_dir) for s in source_configs]
        self.target_dataconfig = DataConfig(target_config, parent_dir)
        
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        try:
            device_id = self.trainer.device_ids[self.trainer.local_rank]
        
            world_size = self.trainer.world_size
            print(f"device id {device_id}, local rank {self.trainer.local_rank}, global rank {self.trainer.global_rank} in world {world_size}")
        except AttributeError:
            print(f"Error when trying to fetch device and rank info")
            print(f"Assume dataset is being setup without a trainer, set device id as 0, global rank as 0, world size as 1")
            device_id = 0
            world_size = 1
        self.batch_size_per_rank = int(self.batch_size/world_size)

        def assign_domain_label(label, domain_idx):
            return np.full_like(label, domain_idx)

        self.train_r_s_b = [dc.get_datapipe_by_subset('single_train_readcount_bound', resample=True) for dc in self.source_dataconfigs]
        self.train_r_s_ub = [dc.get_datapipe_by_subset('single_train_readcount_unbound', resample=True) for dc in self.source_dataconfigs]
        self.train_r_s = wds.RandomMix([*self.train_r_s_b, *self.train_r_s_ub])
        self.val_r_s = ChainDataset([dc.get_datapipe_by_subset('single_val_readcount', resample=False) for dc in self.source_dataconfigs])
        self.test_r_s = ChainDataset([dc.get_datapipe_by_subset('single_test_readcount', resample=False, keep_key=True) for dc in self.source_dataconfigs])

        self.train_d_s = wds.RandomMix([dc.get_datapipe_by_subset('single_train_domain', resample=True, transforms={'label': partial(assign_domain_label, domain_idx=0)}) 
                                        for dc in self.source_dataconfigs])
        self.train_d_t = self.target_dataconfig.get_datapipe_by_subset('single_train_domain', resample=True, transforms={'label': partial(assign_domain_label, domain_idx=1)})

    def train_dataloader(self):
        prefetch_factor = 32
        num_workers = max(int(self.num_workers/3), 1)
            
        train_r_s_loader = wds.WebLoader(self.train_r_s, num_workers=num_workers,
                                       batch_size=self.batch_size_per_rank,
                                       pin_memory=True, prefetch_factor=prefetch_factor).with_epoch(self.steps_per_epoch)
        train_d_s_loader = wds.WebLoader(self.train_d_s, num_workers=num_workers,
                                       batch_size=self.batch_size_per_rank,
                                       pin_memory=True, prefetch_factor=prefetch_factor).with_epoch(self.steps_per_epoch)
        train_d_t_loader = wds.WebLoader(self.train_d_t, num_workers=num_workers,
                                       batch_size=self.batch_size_per_rank,
                                       pin_memory=True, prefetch_factor=prefetch_factor).with_epoch(self.steps_per_epoch)

        return CombinedLoader({'train_readcount_source': train_r_s_loader,
                               'domain_source': train_d_s_loader, 
                               'domain_target': train_d_t_loader})

    def val_dataloader(self):
        return wds.WebLoader(self.val_r_s, num_workers=self.num_workers, batch_size=self.batch_size_per_rank, pin_memory=True)

    def test_dataloader(self):
        return wds.WebLoader(self.test_r_s, num_workers=self.num_workers, 
                              batch_size=self.batch_size_per_rank, 
                              collate_fn=collate_add_domain_source, pin_memory=True)

class ADDADataModuleWds(pl.LightningDataModule):
    def __init__(self,
                 source_configs:list,
                 target_config:str,
                 parent_dir="./",
                 num_workers=1, 
                 batch_size=512, steps_per_epoch=1000):
        super().__init__()
        
        self.source_dataconfigs = [DataConfig(s, parent_dir) for s in source_configs]
        self.target_dataconfig = DataConfig(target_config, parent_dir)
        
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        try:
            device_id = self.trainer.device_ids[self.trainer.local_rank]
        
            world_size = self.trainer.world_size
            print(f"device id {device_id}, local rank {self.trainer.local_rank}, global rank {self.trainer.global_rank} in world {world_size}")
        except AttributeError:
            print(f"Error when trying to fetch device and rank info")
            print(f"Assume dataset is being setup without a trainer, set device id as 0, global rank as 0, world size as 1")
            device_id = 0
            world_size = 1
        self.batch_size_per_rank = int(self.batch_size/world_size)

        self.train_d_s = wds.RandomMix([dc.get_datapipe_by_subset('single_train_domain', resample=True) 
                                        for dc in self.source_dataconfigs])
        self.train_d_t = self.target_dataconfig.get_datapipe_by_subset('single_train_domain', resample=True)
        self.val_r_t = self.target_dataconfig.get_datapipe_by_subset('single_val_readcount', resample=False)

    def train_dataloader(self):
        prefetch_factor = 32
        num_workers = max(int(self.num_workers/4), 1)
            
        train_d_s_loader = wds.WebLoader(self.train_d_s, num_workers=num_workers,
                                       batch_size=self.batch_size_per_rank,
                                       pin_memory=True, prefetch_factor=prefetch_factor).with_epoch(self.steps_per_epoch)
        train_d_t_loader = wds.WebLoader(self.train_d_t, num_workers=num_workers,
                                       batch_size=self.batch_size_per_rank,
                                       pin_memory=True, prefetch_factor=prefetch_factor).with_epoch(self.steps_per_epoch)

        return CombinedLoader({'domain_source': train_d_s_loader, 
                               'domain_target': train_d_t_loader})

    def val_dataloader(self):
        return wds.WebLoader(self.val_r_t, num_workers=self.num_workers, batch_size=self.batch_size_per_rank, pin_memory=True)


class ADDADataModuleWdsACC(pl.LightningDataModule):
    def __init__(self,
                 source_configs:list,
                 target_config:str,
                 parent_dir="./",
                 num_workers=1, 
                 batch_size=512, steps_per_epoch=1000):
        super().__init__()
        
        self.source_dataconfigs = [DataConfig(s, parent_dir) for s in source_configs]
        self.target_dataconfig = DataConfig(target_config, parent_dir)
        
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        try:
            device_id = self.trainer.device_ids[self.trainer.local_rank]
        
            world_size = self.trainer.world_size
            print(f"device id {device_id}, local rank {self.trainer.local_rank}, global rank {self.trainer.global_rank} in world {world_size}")
        except AttributeError:
            print(f"Error when trying to fetch device and rank info")
            print(f"Assume dataset is being setup without a trainer, set device id as 0, global rank as 0, world size as 1")
            device_id = 0
            world_size = 1
        self.batch_size_per_rank = int(self.batch_size/world_size)
        self.num_workers_per_loader = max(int(self.num_workers/2), 1)

        self.train_d_s_acc = wds.RandomMix([dc.get_datapipe('single_train_domain', 'accessible_for_domain_task', resample=True)
                                        for dc in self.source_dataconfigs])
        self.train_d_s_inacc = wds.RandomMix([dc.get_datapipe('single_train_domain', 'inaccessible_for_domain_task', resample=True) 
                                        for dc in self.source_dataconfigs])
        self.train_d_t_acc = self.target_dataconfig.get_datapipe('single_train_domain', 'accessible_for_domain_task', resample=True)
        self.train_d_t_inacc = self.target_dataconfig.get_datapipe('single_train_domain', 'inaccessible_for_domain_task', resample=True)
        self.val_r_t = self.target_dataconfig.get_datapipe_by_subset('single_val_readcount', resample=False)

    def train_dataloader(self):
        prefetch_factor = 32
        
        steps_per_loader = int(self.steps_per_epoch/2)
        train_d_s_acc_loader = wds.WebLoader(self.train_d_s_acc, num_workers=self.num_workers_per_loader,
                                         batch_size=self.batch_size_per_rank,
                                         pin_memory=True, prefetch_factor=prefetch_factor).with_epoch(steps_per_loader)
        train_d_s_inacc_loader = wds.WebLoader(self.train_d_s_inacc, num_workers=self.num_workers_per_loader,
                                         batch_size=self.batch_size_per_rank,
                                         pin_memory=True, prefetch_factor=prefetch_factor).with_epoch(steps_per_loader)

        train_d_t_acc_loader = wds.WebLoader(self.train_d_t_acc, num_workers=self.num_workers_per_loader,
                                          batch_size=self.batch_size_per_rank,
                                          pin_memory=True, prefetch_factor=prefetch_factor).with_epoch(steps_per_loader)
        train_d_t_inacc_loader = wds.WebLoader(self.train_d_t_inacc, num_workers=self.num_workers_per_loader,
                                          batch_size=self.batch_size_per_rank,
                                          pin_memory=True, prefetch_factor=prefetch_factor).with_epoch(steps_per_loader)

        return CombinedLoader({'domain_source_acc': train_d_s_acc_loader, 
                               'domain_source_inacc': train_d_s_inacc_loader, 
                               'domain_target_acc': train_d_t_acc_loader,
                               'domain_target_inacc': train_d_t_inacc_loader})

    def val_dataloader(self):
        return wds.WebLoader(self.val_r_t, num_workers=self.num_workers, batch_size=self.batch_size_per_rank, pin_memory=True)

def center_and_expand_df(df, window):
    "Center given dataframe regions and expand to specified window length"
    halfR = int(window/2)
    df = df.assign(mid = lambda x: ((x['start'] + x['end'])/2).astype(int)).assign(start=lambda x: x['mid']-halfR,
                                                                                   end=lambda x: x['mid']+halfR)
    return df[['chrom', 'start', 'end']]

class BedDataModule(pl.LightningDataModule):
    "A data module takes bed files as training regions, only take first 3 columns as chrom, start, end, and will take both strands"
    def __init__(self, config_file,
                  batch_size=128, num_workers=0):
        super().__init__()
        self.config = safe_load(open(config_file))
        self.pos_bed = self.config["pos_bed"]
        self.neg_bed = self.config["neg_bed"]
        self.genome_fasta_file = self.config["genome_fasta_file"]
        self.genome_size_file = self.config["genome_size_file"]
        self.bigwigs = self.config["pre_bws"]

        self.input_window_length = self.config["input_window_length"]
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # load bed files
        self.pos_df = pd.read_table(self.pos_bed, header=None, usecols=range(3), names=["chrom", "start", "end"])
        self.pos_df = center_and_expand_df(self.pos_df, self.input_window_length)
        if self.neg_bed is not None:
            self.neg_df = pd.read_table(self.neg_bed, header=None, usecols=range(3), names=["chrom", "start", "end"])
            self.neg_df = center_and_expand_df(self.neg_df, self.input_window_length)
        else:
            self.neg_df = scl.random_coords(gs=self.genome_size_file, l=self.input_window_length, n=len(self.pos_df)*5) # oversample the negative random regions

        # assign labels and strand
        self.pos_df['label'] = 1
        self.neg_df['label'] = 0
        self.pos_df = pd.concat([self.pos_df.assign(strand='+'), self.pos_df.assign(strand='-')], axis=0)
        self.neg_df = pd.concat([self.neg_df.assign(strand='+'), self.neg_df.assign(strand='-')], axis=0)

        # split into train, val, test
        from sklearn.model_selection import train_test_split
        self.pos_train_df, self.pos_val_test_df = train_test_split(self.pos_df, test_size=0.2)
        self.pos_val_df, self.pos_test_df = train_test_split(self.pos_val_test_df, test_size=0.5)

        self.neg_train_df, self.neg_val_test_df = train_test_split(self.neg_df, test_size=0.2)
        self.neg_val_df, self.neg_test_df = train_test_split(self.neg_val_test_df, test_size=0.5)

        # stitch and shuffle
        self.train_df = pd.concat([self.pos_train_df, self.neg_train_df], axis=0).sample(frac=1.)
        self.val_df = pd.concat([self.pos_val_df, self.neg_val_df], axis=0).sample(frac=1.)
        self.test_df = pd.concat([self.pos_test_df, self.neg_test_df], axis=0).sample(frac=1.)

    def setup(self, stage=None):
        try:
            device_id = self.trainer.device_ids[self.trainer.local_rank]
            world_size = self.trainer.world_size
            print(f"device id {device_id}, local rank {self.trainer.local_rank}, global rank {self.trainer.global_rank} in world {world_size}")
        except AttributeError:
            print(f"Error when trying to fetch device and rank info")
            print(f"Assume dataset is being setup without a trainer, set device id as 0, global rank as 0, world size as 1")
            device_id = 0
            world_size = 1
        self.batch_size_per_rank = int(self.batch_size/world_size)
        self.num_workers_per_loader = max(int(self.num_workers/2), 1)

        # create dataloader for each dataset
        from dataset import get_mean_and_std, default_chroms_transform, default_label_transform
        transforms = {"label": default_label_transform}
        if self.bigwigs is not None:
            bws_mean, bws_std = get_mean_and_std(self.bigwigs)
            transforms['chrom'] = partial(default_chroms_transform, mean=bws_mean, std=bws_std)

        # split the dataframes across devices
        train_df_chunk = np.array_split(self.train_df, world_size)[device_id]
        val_df_chunk = np.array_split(self.val_df, world_size)[device_id]
        test_df_chunk = np.array_split(self.test_df, world_size)[device_id]

        del scl.loader._SeqChromDatasetByDataFrame.__len__

        self.train_dl = scl.loader._SeqChromDatasetByDataFrame(train_df_chunk, 
                                                               genome_fasta=self.genome_fasta_file, 
                                                               bigwig_filelist=self.bigwigs, 
                                                               transforms=transforms, 
                                                               return_region=False,
                                                               patch_left=0, patch_right=0,
                                                               shuffle=False)

        self.val_dl = scl.loader._SeqChromDatasetByDataFrame(val_df_chunk, 
                                                             genome_fasta=self.genome_fasta_file, 
                                                             bigwig_filelist=self.bigwigs, 
                                                             transforms=transforms, 
                                                             return_region=False,
                                                             patch_left=0, patch_right=0,
                                                             shuffle=False)

        self.test_dl = scl.loader._SeqChromDatasetByDataFrame(test_df_chunk, 
                                                              genome_fasta=self.genome_fasta_file, 
                                                              bigwig_filelist=self.bigwigs, 
                                                              transforms=transforms, 
                                                              return_region=True,
                                                              patch_left=0, patch_right=0,
                                                              shuffle=False)

    def train_dataloader(self):
        return DataLoader(self.train_dl, batch_size=self.batch_size_per_rank, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dl, batch_size=self.batch_size_per_rank, num_workers=self.num_workers)

    def test_dataloader(self):
        return [DataLoader(self.test_dl, batch_size=self.batch_size_per_rank, num_workers=self.num_workers,
                           collate_fn=collate_add_domain_source),]

