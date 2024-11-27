# copyed train_croppped.py
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import multiprocessing as mp
import platform
import time
import warnings
import pickle
import random

import cv2
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import get_git_hash
from mmdet.utils import collect_env

import mmfewshot  # noqa: F401, F403
from mmfewshot import __version__
# from mmfewshot.detection.apis import train_detector
from mmfewshot.detection.apis.train import train_detector
from mmfewshot.detection.datasets import build_dataset, build_unlabeled_dataset_lloss
from mmfewshot.detection.models import build_detector
from mmfewshot.utils import get_root_logger

from mmfewshot.detection.datasets import (build_dataloader, build_dataset,
                                          get_copy_dataset_type)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a FewShot model')
    parser.add_argument('config', help='train config file path')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    ###################### SET THE Parameters ######################
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--sampling_num', type=int, default=4000)
    parser.add_argument('--select_mode', type=str, default='coreset')
    parser.add_argument('--few_shot_save_path', type=str, default='/home/yjhwang/workspace/FSOD/VFA-Darkdata/data/few_shot_ann_JB/voc/')
    parser.add_argument('--num_of_k', type=int, default=10)
    ###################### SET THE Parameters ######################
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value '
        'to be overwritten is a list, it should be like key="[a,b]" or '
        'key=a,b It also allows nested list/tuple values, e.g. '
        'key="[(a,b),(c,d)]" Note that the quotation marks are necessary '
        'and that no white space is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def setup_multi_processes(cfg):
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        mp.set_start_method(mp_start_method)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if ('OMP_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1):
        omp_num_threads = 1
        warnings.warn(
            f'Setting OMP_NUM_THREADS environment variable for each process '
            f'to be {omp_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'Setting MKL_NUM_THREADS environment variable for each process '
            f'to be {mkl_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    random.seed(args.seed)
    unlabeled_datasets = [
        build_unlabeled_dataset_lloss(
            cfg.data.unlabeled
            )
    ]

    # Get the CLIP inference results
    inference_save_path = 'crane_backbone_features_novel-clip-mobile-crane.pickle'
    if os.path.exists(inference_save_path):
        with open(inference_save_path, mode="rb") as fp:
            crane_features = pickle.load(fp)
    else:
        from mmdet.apis.data_select import single_gpu_get_clip_crane
        crane_features = single_gpu_get_clip_crane(unlabeled_datasets[0])
        with open(inference_save_path, mode="wb") as fp:
            pickle.dump(crane_features, fp)

    ##################  STAGE 1  ##################
    # Sort the results (get data with high similarity to class text)
    SAMPLING_NUM = args.sampling_num
    sorted_by_l2_distance = sorted(crane_features, key=lambda x: x["l2"])
    sampling_data = {i['filename']:i["feat"] for i in sorted_by_l2_distance[:SAMPLING_NUM]}
    sampling_feats = torch.stack([v for k, v in sampling_data.items()])
    sampling_paths = [k for k, v in sampling_data.items()]

    ##################  STAGE 2  ##################
    selected_name = []
    if args.select_mode == 'coreset':
        from utils.kcenterGreedy import kCenterGreedy
        sampling = kCenterGreedy(sampling_feats)  
        activeSet = sampling.select_batch_([], args.num_of_k)
    elif args.select_mode == 'random':
        activeSet = list(range(len(sampling_paths)))
        random.shuffle(activeSet)
        activeSet = activeSet[:args.num_of_k]
    print(" *** Selected Results :")
    for sel_i in activeSet:
        selected_name.append(sampling_paths[sel_i])
        print(sampling_paths[sel_i])
    
    save_dir = os.path.join(args.few_shot_save_path, f'benchmark_{args.num_of_k}shot')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, 'box_10shot_crane_train.txt')
    with open(filepath, 'w') as fp:
        for name_i in selected_name:
            fp.write(f'{name_i} 0\n')
    print(f"\n ** Saved the file in {filepath}")


if __name__ == '__main__':
    main()
