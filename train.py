#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
"""
import argparse
import logging
import os
import time
from collections import OrderedDict
from collections import Counter
from contextlib import suppress
from datetime import datetime
from functools import partial
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torchvision.utils
import yaml
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config, Mixup, FastCollateMixup, AugMixDataset
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, \
    LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, \
    convert_splitbn_model, convert_sync_batchnorm, model_parameters, set_fast_norm
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler

import datasets

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


_logger = logging.getLogger('train')

# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside of the dataset group because it is positional.
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
group.add_argument('--train-split', metavar='NAME', default='train',
                    help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                    help='dataset validation split (default: validation)')
group.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                    help='Name of model to train (default: "resnet50")')
group.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                    help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                    help='Image size (default: None => model default)')
group.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
group.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                    help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                    help='Validation batch size override (default: None)')
group.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
group.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
group.add_argument('--grad-checkpointing', action='store_true', default=False,
                    help='Enable gradient checkpointing through model blocks/stages')
group.add_argument('--fast-norm', default=False, action='store_true',
                    help='enable experimental fast-norm')

scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                             help="Enable AOT Autograd support.")

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd")')
group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                    help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                    help='Clip gradient norm (default: None, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,
                    help='layer-wise learning rate decay (default: None)')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
group.add_argument('--sched-on-updates', action='store_true', default=False,
                    help='Apply LR scheduler step on update instead of epoch end.')
group.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                    help='base learning rate: lr = lr_base * global_batch_size / base_size')
group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                    help='base learning rate batch size (divisor, default: 256).')
group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                    help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
group.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                    help='warmup learning rate (default: 1e-5)')
group.add_argument('--min-lr', type=float, default=0, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                    help='number of epochs to train (default: 300)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                    help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
group.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                    help='list of decay epoch indices for multistep lr. must be increasing')
group.add_argument('--decay-epochs', type=float, default=90, metavar='N',
                    help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
group.add_argument('--warmup-prefix', action='store_true', default=False,
                    help='Exclude warmup period from decay schedule.'),
group.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                    help='patience epochs for Plateau LR scheduler (default: 10)')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                    help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                    help='Disable all training augmentation, override other train aug args')
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                    help='Random resize scale (default: 0.08 1.0)')
group.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                    help='Random resize aspect ratio (default: 0.75 1.33)')
group.add_argument('--hflip', type=float, default=0.5,
                    help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                    help='Vertical flip training aug probability')
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                    help='Color jitter factor (default: 0.4)')
group.add_argument('--aa', type=str, default=None, metavar='NAME',
                    help='Use AutoAugment policy. "v0" or "original". (default: None)'),
group.add_argument('--aug-repeats', type=float, default=0,
                    help='Number of augmentation repetitions (distributed training only) (default: 0)')
group.add_argument('--aug-splits', type=int, default=0,
                    help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
group.add_argument('--jsd-loss', action='store_true', default=False,
                    help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
group.add_argument('--bce-loss', action='store_true', default=False,
                    help='Enable BCE loss w/ Mixup/CutMix use.')
group.add_argument('--bce-target-thresh', type=float, default=None,
                    help='Threshold for binarizing softened BCE targets (default: None, disabled)')
group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                    help='Random erase prob (default: 0.)')
group.add_argument('--remode', type=str, default='pixel',
                    help='Random erase mode (default: "pixel")')
group.add_argument('--recount', type=int, default=1,
                    help='Random erase count (default: 1)')
group.add_argument('--resplit', action='store_true', default=False,
                    help='Do not random erase first (clean) augmentation split')
group.add_argument('--mixup', type=float, default=0.0,
                    help='mixup alpha, mixup enabled if > 0. (default: 0.)')
group.add_argument('--cutmix', type=float, default=0.0,
                    help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                    help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
group.add_argument('--mixup-prob', type=float, default=1.0,
                    help='Probability of performing mixup or cutmix when either/both is enabled')
group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                    help='Probability of switching to cutmix when both mixup and cutmix enabled')
group.add_argument('--mixup-mode', type=str, default='batch',
                    help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                    help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
group.add_argument('--smoothing', type=float, default=0.1,
                    help='Label smoothing (default: 0.1)')
group.add_argument('--train-interpolation', type=str, default='random',
                    help='Training interpolation (random, bilinear, bicubic default: "random")')
group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.)')
group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                    help='Drop path rate (default: None)')
group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
group.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
group.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
group.add_argument('--sync-bn', action='store_true',
                    help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
group.add_argument('--dist-bn', type=str, default='reduce',
                    help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
group.add_argument('--split-bn', action='store_true',
                    help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
group = parser.add_argument_group('Model exponential moving average parameters')
group.add_argument('--model-ema', action='store_true', default=False,
                    help='Enable tracking moving average of model weights')
group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                    help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
group.add_argument('--model-ema-decay', type=float, default=0.9998,
                    help='decay factor for model weights moving average (default: 0.9998)')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
group.add_argument('--worker-seeding', type=str, default='all',
                    help='worker seed mode (default: all)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                    help='number of checkpoints to keep (default: 10)')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
group.add_argument('--save-images', action='store_true', default=False,
                    help='save images of input bathes every log interval for debugging')
group.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
group.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
group.add_argument('--amp-impl', default='native', type=str,
                    help='AMP impl to use, "native" or "apex" (default: native)')
group.add_argument('--no-ddp-bb', action='store_true', default=False,
                    help='Force broadcast buffers for native DDP to off.')
group.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
group.add_argument('--output', default='', type=str, metavar='PATH',
                    help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                    help='name of train experiment, name of sub-folder for output')
group.add_argument('--subexperiment', default='', type=str, metavar='NAME',
                    help='name of train subexperiment, name of sub-sub-folder for output')
group.add_argument('--group', default='', type=str, metavar='NAME',
                    help='name of train group for DDP visualization with wandb')
group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                    help='Best metric (default: "top1"')
group.add_argument('--tta', type=int, default=0, metavar='N',
                    help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                    help='use the multi-epochs-loader to save time at the beginning of every epoch')
group.add_argument('--log-wandb', action='store_true', default=False,
                    help='log training and validation metrics to wandb')
# GipMed
group.add_argument('--no-grad', action='store_true', default=False,
                    help='Enable fine tuning only.'),
group.add_argument('--num-output', type=int, default=None, metavar='N',
                    help='number of label classes in new head'),
group.add_argument('-balsam', '--balanced_sampling', dest='balanced_sampling', action='store_true', help='balanced_sampling')

# ========================
parser.add_argument('-tf', '--test_fold', default=1, type=int, help='fold to be as VALIDATION FOLD, if -1 there is no validation. refered to as TEST FOLD in folder hiererchy and code. very confusing, I agree.')
parser.add_argument('-d', dest='dx', action='store_true', help='Use ONLY DX cut slides')
parser.add_argument('-time', dest='time', action='store_true', help='save train timing data ?')
parser.add_argument('-tar', '--target', default='Survival_Time', type=str, help='label: Her2/ER/PR/EGFR/PDL1')
parser.add_argument('--n_patches_test', default=1, type=int, help='# of patches at test time')
parser.add_argument('--n_patches_train', default=10, type=int, help='# of patches at train time')
# parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--transform_type', default='rvf', type=str, help='none / flip / wcfrs (weak color+flip+rotate+scale)')
parser.add_argument('--bootstrap', action='store_true', help='use bootstrap to estimate test AUC error')
parser.add_argument('--eval_rate', type=int, default=5, help='Evaluate validation set every # epochs')
parser.add_argument('--c_param', default=0.1, type=float, help='color jitter parameter')
parser.add_argument('-im', dest='images', action='store_true', help='save data images?')
parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches')
parser.add_argument('--loan', action='store_true', help='Localized Annotation for strongly supervised training')
parser.add_argument('--er_eq_pr', action='store_true', help='while training, take only er=pr examples')
parser.add_argument('--focal', action='store_true', help='use focal loss with gamma=2')
parser.add_argument('--slide_per_block', action='store_true', help='for carmel, take only one slide per block')
parser.add_argument('-baldat', '--balanced_dataset', dest='balanced_dataset', action='store_true', help='take same # of positive and negative patients from each dataset')
parser.add_argument('--RAM_saver', action='store_true', help='use only a quarter of the slides + reshuffle every 100 epochs')
parser.add_argument('-tl', '--transfer_learning', default='', type=str, help='use model trained on another experiment')
parser.add_argument('-nt', '--num_tiles', type=int, default=500, help='Number of tiles per slide')
parser.add_argument('-tpi', '--tiles_per_iter', type=int, default=500, help='Number of tiles per batch')

# args.folds = list(map(int, args.folds[0]))
# End GipMed
parser.add_argument('--supervised', action='store_true', help='work only with the test fold as 80/20 partition')
parser.add_argument('-ef','--extract_features', action='store_true', help='feature extractor')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    # Tile size definition:
    TILE_SIZE = 256

    utils.setup_default_logging()
    args, args_text = _parse_args()

    # Tile size definition:
    if (args.dataset[:3] == 'TMA') and (args.mag == 7):
        TILE_SIZE = 512

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if args.data and not args.data_dir:
        args.data_dir = args.data
    args.prefetcher = not args.no_prefetcher
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    # if utils.is_primary(args) and args.log_wandb:
    #     if has_wandb:
    #         wandb.init(entity="gipmed-vit", project=args.experiment, config=args, group="DDP")
    #     else:
    #         _logger.warning(
    #             "You've requested to log metrics to wandb but package not found. "
    #             "Metrics not being logged to wandb, try `pip install wandb`")
    if args.group:
        run = wandb.init(entity="gipmed-vit", project=args.experiment, config=args, group=args.group)
    else:
        run = wandb.init(entity="gipmed-vit", project=args.experiment, config=args)

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            use_amp = 'apex'
            assert args.amp_dtype == 'float16'
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            use_amp = 'native'
            assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    print('args.test_fold:', args.test_fold)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
    )

    # Tom
    if args.no_grad:
        for p in model.parameters():
            p.requires_grad = False
        for p in model.head.parameters():
            p.requires_grad = True
    # Tom end

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))
    
    # Tom Start
    if args.extract_features:
        print(model)
        model.head = nn.Identity()
        print(model)
    # Tom End

    # move model to GPU, enable channels last layout if set
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ''  # disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == 'apex':
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)
    elif args.torchcompile:
        # FIXME dynamo might need move below DDP wrapping? TBD
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        torch._dynamo.reset()
        model = torch.compile(model, backend=args.torchcompile)
    elif args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
                f'and global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.')

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == 'apex':
        assert device.type == 'cuda'
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
    elif use_amp == 'native':
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type == 'cuda':
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            _logger.info('AMP not enabled. Training in float32.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == 'apex':
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    # create the train and eval datasets
    # dataset_train = create_dataset(
    #     args.dataset,
    #     root=args.data_dir,
    #     split=args.train_split,
    #     is_training=True,
    #     class_map={'Positive':1, 'Negative':0},
    #     # class_map=args.class_map,
    #     download=args.dataset_download,
    #     batch_size=args.batch_size,
    #     seed=args.seed,
    #     repeats=args.epoch_repeats,
    # )

    # dataset_eval = create_dataset(
    #     args.dataset,
    #     root=args.data_dir,
    #     split=args.val_split,
    #     is_training=False,
    #     class_map={'Positive': 1, 'Negative': 0},
    #     # class_map=args.class_map,
    #     download=args.dataset_download,
    #     batch_size=args.batch_size,
    # )

    # Get data:
    if not args.supervised:
        train_dset = datasets.WSI_REGdataset(DataSet=args.dataset,
                                             tile_size=TILE_SIZE,
                                             target_kind=args.target,
                                             test_fold=args.test_fold,
                                             train=True,
                                             print_timing=args.time,
                                             transform_type=args.transform_type,
                                             n_tiles=args.n_patches_train,
                                             color_param=args.c_param,
                                             get_images=args.images,
                                             desired_slide_magnification=args.mag,
                                             DX=args.dx,
                                             loan=args.loan,
                                             er_eq_pr=args.er_eq_pr,
                                             slide_per_block=args.slide_per_block,
                                             balanced_dataset=args.balanced_dataset,
                                             RAM_saver=args.RAM_saver
                                             )
    if not args.extract_features:
        test_dset = datasets.WSI_REGdataset(DataSet=args.dataset,
                                            tile_size=TILE_SIZE,
                                            target_kind=args.target,
                                            test_fold=args.test_fold,
                                            train=False,
                                            print_timing=False,
                                            transform_type='none',
                                            n_tiles=args.n_patches_test,
                                            get_images=args.images,
                                            desired_slide_magnification=args.mag,
                                            DX=args.dx,
                                            loan=args.loan,
                                            er_eq_pr=args.er_eq_pr,
                                            RAM_saver=args.RAM_saver
                                            )

    inf_dset = datasets.Infer_Dataset(DataSet=args.dataset,
                                      tile_size=TILE_SIZE,
                                      tiles_per_iter=args.tiles_per_iter,
                                      target_kind=args.target,
                                      # folds=args.folds,
                                      num_tiles=args.num_tiles,
                                      desired_slide_magnification=args.mag,
                                      dx=args.dx)
                                      # resume_slide=slide_num,
                                      # patch_dir=args.patch_dir,
                                      # chosen_seed=args.seed)

    NUM_SLIDES = len(inf_dset.image_file_names)
    sampler = None
    do_shuffle = True

    if args.supervised:
        temp_size = len(test_dset)
        train_dset, test_dset = random_split(test_dset, (int(temp_size*0.8), temp_size - int(temp_size*0.8)))

    if args.balanced_sampling:
        labels = pd.DataFrame(train_dset.dataset.target * train_dset.dataset.factor)
        n_pos = np.sum(labels == 'Positive').item()
        n_neg = np.sum(labels == 'Negative').item()
        weights = pd.DataFrame(np.zeros(n_pos+n_neg))
        print("train size:" + str(len(train_dset)))
        print("test size:" + str(len(test_dset)))
        print("weights df shape is" + str(weights.shape))
        print(weights[np.array(labels == 'Positive')])
        weights[np.array(labels == 'Positive')] = 1 / n_pos
        weights[np.array(labels == 'Negative')] = 1 / n_neg
        do_shuffle = False  # the sampler shuffles
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weights=weights.squeeze(), num_samples=len(train_dset))
    loader_train = DataLoader(train_dset, batch_size=args.batch_size, shuffle=do_shuffle, num_workers=args.workers, pin_memory=True, sampler=sampler)

    inf_loader = DataLoader(inf_dset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    new_slide = True

    # loaders prints
    # print("train loader:")
    # print(loader_train)
    # print("train loader enumerate:")
    # print(list(enumerate(loader_train))[0])

    eval_workers = args.workers
    if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
        # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
        eval_workers = min(2, args.workers)
    if not args.extract_features:
        loader_eval = DataLoader(test_dset, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=args.workers, pin_memory=True)

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if num_aug_splits > 1:
        train_dset = AugMixDataset(train_dset, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    # train_interpolation = args.train_interpolation
    # if args.no_aug or not train_interpolation:
    #     train_interpolation = data_config['interpolation']
    # timm_loader_train = create_loader(
    #     dataset_train,
    #     input_size=data_config['input_size'],
    #     batch_size=args.batch_size,
    #     is_training=True,
    #     use_prefetcher=args.prefetcher,
    #     no_aug=args.no_aug,
    #     re_prob=args.reprob,
    #     re_mode=args.remode,
    #     re_count=args.recount,
    #     re_split=args.resplit,
    #     scale=args.scale,
    #     ratio=args.ratio,
    #     hflip=args.hflip,
    #     vflip=args.vflip,
    #     color_jitter=args.color_jitter,
    #     auto_augment=args.aa,
    #     num_aug_repeats=args.aug_repeats,
    #     num_aug_splits=num_aug_splits,
    #     interpolation=train_interpolation,
    #     mean=data_config['mean'],
    #     std=data_config['std'],
    #     num_workers=args.workers,
    #     distributed=args.distributed,
    #     collate_fn=collate_fn,
    #     pin_memory=args.pin_mem,
    #     device=device,
    #     use_multi_epochs_loader=args.use_multi_epochs_loader,
    #     worker_seeding=args.worker_seeding,
    # )

    # loader_eval = create_loader(
    #     dataset_eval,
    #     input_size=data_config['input_size'],
    #     batch_size=args.validation_batch_size or args.batch_size,
    #     is_training=False,
    #     use_prefetcher=args.prefetcher,
    #     interpolation=data_config['interpolation'],
    #     mean=data_config['mean'],
    #     std=data_config['std'],
    #     num_workers=eval_workers,
    #     distributed=args.distributed,
    #     crop_pct=data_config['crop_pct'],
    #     pin_memory=args.pin_mem,
    #     device=device,
    # )

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
            if args.subexperiment:
                subexp_name = args.subexperiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name, subexp_name)
        decreasing = True if eval_metric == 'loss' else False
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    # setup learning rate schedule and starting epoch
    updates_per_epoch = len(loader_train)
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        _logger.info(
            f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')

    try:
        for epoch in range(start_epoch, num_epochs):
            if not args.extract_features:
                if hasattr(train_dset, 'set_epoch'):
                    train_dset.set_epoch(epoch)
                elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                    loader_train.sampler.set_epoch(epoch)
                train_metrics = train_one_epoch(
                    epoch,
                    model,
                    loader_train,
                    optimizer,
                    train_loss_fn,
                    args,
                    run,
                    lr_scheduler=lr_scheduler,
                    saver=saver,
                    output_dir=output_dir,
                    amp_autocast=amp_autocast,
                    loss_scaler=loss_scaler,
                    model_ema=model_ema,
                    mixup_fn=mixup_fn
                    )

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if utils.is_primary(args):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')
            inf_dset.reset_counter()
            eval_metrics = validate(
                model,
                inf_loader,
                train_loss_fn,
                # validate_loss_fn,
                args,
                run,
                amp_autocast=amp_autocast,
            )

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                    utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                ema_eval_metrics = validate(
                    model_ema.module,
                    inf_loader,
                    validate_loss_fn,
                    args,
                    run,
                    amp_autocast=amp_autocast,
                    log_suffix=' (EMA)',
                )
                eval_metrics = ema_eval_metrics

            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    # if has_wandb:
    #     wandb.run.finish()

def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        args,
        run,
        device=torch.device('cuda'),
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=suppress,
        loss_scaler=None,
        model_ema=None,
        mixup_fn=None,
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    auces_per_minibatch_m = utils.AverageMeter()


    model.train()

    end = time.time()
    num_batches_per_epoch = len(loader)
    last_idx = num_batches_per_epoch - 1
    num_updates = epoch * num_batches_per_epoch
    # for batch_idx, (input, target) in enumerate(loader):

    for batch_idx, minibatch in enumerate(loader):
        input = minibatch['Data']
        target = minibatch['Target']
        # f_names = minibatch['File Names']
        # slide_names_batch = [os.path.basename(f_name) for f_name in f_names]
        # slide_names.extend(slide_names_batch)
        input = input.to(device)
        target = target.to(device)
        last_batch = batch_idx == last_idx
        data_time_m.update(time.time() - end)

        if not args.prefetcher:
            input, target = input.to(device), target.to(device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        with amp_autocast():
            output = model(input)
            output = torch.nn.functional.softmax(output, dim=1)

            # print("target shape:" + str(target.shape))
            # print("target:" + str(target))
            #
            # print("output shape:" + str(output.shape))
            # print("output:" + str(output))
            loss = loss_fn(output, target)
            roc_auc = roc_auc_score(target.cpu().detach(), output.cpu().detach()[:, 1])
            auc = roc_auc if roc_auc.size == 1 else roc_auc[0]

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
            auces_per_minibatch_m.update(auc.item(), input.size(0))

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad,
                clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order
            )
        else:
            loss.backward(create_graph=second_order)
            if args.clip_grad is not None:
                utils.dispatch_clip_grad(
                    model_parameters(model, exclude_head='agc' in args.clip_mode),
                    value=args.clip_grad,
                    mode=args.clip_mode
                )
            optimizer.step()

        if model_ema is not None:
            model_ema.update(model)

        torch.cuda.synchronize()

        num_updates += 1
        batch_time_m.update(time.time() - end)
        if last_batch or batch_idx % args.log_interval == 0:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item(), input.size(0))

            if utils.is_primary(args):
                _logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '
                    'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * args.world_size / batch_time_m.val,
                        rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                        lr=lr,
                        data_time=data_time_m)
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True
                    )
            # ROC
            run.log({"roc_train": wandb.plot.roc_curve(target.cpu().detach(), output.cpu().detach())})
            # Precision-Recall
            run.log({"pr_train": wandb.plot.pr_curve(target.cpu().detach(), output.cpu().detach())})
            # Confusion Matrices
            roc_auc_train = roc_auc_score(target.cpu().detach(), output.cpu().detach()[:, 1])
            run.log({"old_auc_train": roc_auc_train if roc_auc_train.size == 1 else roc_auc_train[0]})
            # =========================================================================================

            if saver is not None and args.recovery_interval and (
                    last_batch or (batch_idx + 1) % args.recovery_interval == 0):
                saver.save_recovery(epoch, batch_idx=batch_idx)

            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            end = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()
    run.log({"auc_train": auces_per_minibatch_m.avg})
    return OrderedDict([('loss', losses_m.avg)])


def validate(
        model,
        loader,
        loss_fn,
        args,
        run,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        log_suffix=''
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    auces_per_patch_m = 0
    # auces_per_minibatch_m = utils.AverageMeter()
    auces_per_slide_m = 0


    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    new_slide = True
    slide_num = 0
    N_classes = 2
    
    all_outputs = np.zeros((1,2))
    all_targets = np.zeros((1,1))

    all_outputs_slide_level = np.zeros((1,2))
    all_targets_slide_level = np.zeros((1,1))




    end = time.time()
    # last_idx = len(loader) - 1
    with torch.no_grad():
        # for batch_idx, (input, target) in enumerate(loader):
        for batch_idx, minibatch in enumerate(loader):
            input = minibatch['Data']
            target = minibatch['Label']
            last_batch = minibatch['Is Last Batch']
            slide_file = minibatch['Slide Filename']
            print(slide_file, slide_file[0])
            slide_dataset = minibatch['Slide DataSet']
            patch_locs = minibatch['Patch Loc']
            # f_names = minibatch['File Names']
            # slide_names_batch = [os.path.basename(f_name) for f_name in f_names]
            # slide_names.extend(slide_names_batch)
            print ("batch_idx", batch_idx)
            print ("target", target)
            print("slide_num", slide_num)

            if new_slide:
                n_tiles = loader.dataset.num_tiles[slide_num]
                if args.extract_features:
                    temp_outputs_slide_level = np.zeros((1,384))
                else:
                    temp_outputs_slide_level = np.zeros((1,2))
                temp_targets_slide_level = np.zeros((1,1))
                slide_batch_num = 0
                new_slide = False

            input = input.squeeze(0)

            input = input.to(device)
            if target.item() == 1:
                target = torch.ones(input.shape[0], 1, dtype=torch.int64)
            else:
                target = torch.zeros(input.shape[0], 1, dtype=torch.int64)
            target = target.to(device)

            # Old
            batch_time_m.update(time.time() - end)

            # for (input, target) in zip(inputs, targets):
            if not args.prefetcher:
                input = input.to(device)
                target = target.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            output = torch.nn.functional.softmax(output, dim=1)

            # augmentation reduction
            reduce_factor = args.tta
            if reduce_factor > 1:
                output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                target = target[0:target.size(0):reduce_factor]

            # print("input shape:" + str(input.shape))
            # print("target shape:" + str(target.shape))
            # print("outpus shape:" + str(output.shape))
            #
            # print("target:" + str(target))
            # print("outpus:" + str(output))


            loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # roc_auc_per_minibatch = roc_auc_score(target.cpu().detach(), output.cpu().detach()[:, 1])
            # auc_per_minibatch = roc_auc_per_minibatch if roc_auc_per_minibatch.size == 1 else roc_auc_per_minibatch[0]

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == 'cuda':
                torch.cuda.synchronize()
            losses_m.update(reduced_loss.item(), input.size(0))

            # auces_per_minibatch_m.update(auc_per_minibatch.item(), input.size(0))

            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

            temp_outputs_slide_level = np.concatenate((temp_outputs_slide_level,output.cpu().detach().numpy()), axis=0)
            temp_targets_slide_level = np.concatenate((temp_targets_slide_level,target.cpu().detach().numpy()), axis=0)

            slide_batch_num += 1

            batch_time_m.update(time.time() - end)
            end = time.time()

            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                # addition for auc per slide calculation
                # all_targets.append(target.cpu().numpy()[0][0])
                if args.extract_features:
                    torch.save(temp_outputs_slide_level, ("./TCGA_500/"+slide_file[0]+'_features.pt'))
                else:
                    temp_outputs_slide_level = temp_outputs_slide_level[1:]
                    temp_targets_slide_level = temp_targets_slide_level[1:]
                    all_outputs = np.concatenate((all_outputs,temp_outputs_slide_level), axis=0)
                    all_targets = np.concatenate((all_targets,temp_targets_slide_level), axis=0)
                    all_outputs_slide_level = np.concatenate((all_outputs_slide_level, np.reshape(temp_outputs_slide_level.mean(0), (1, 2))), axis=0)
                    all_targets_slide_level = np.concatenate((all_targets_slide_level, np.reshape(temp_targets_slide_level[0],(1,1))), axis=0)

                    #original line
                    # predicted = current_slide_tile_scores[model_ind].mean(0).argmax()
            
                    # if N_classes == 2:
                    #     # patch_scores[slide_num, model_ind, :n_tiles] = all_outputs_slide_level[model_ind][:, 1]
                    #     # all_scores[slide_num, model_ind] = all_outputs_slide_level[model_ind][:, 1].mean()
                    #     if target == 1 and predicted == 1:
                    #         correct_pos[model_ind] += 1
                    #     elif target == 0 and predicted == 0:
                    #         correct_neg[model_ind] += 1

                    #Old
                    log_name = 'Test' + log_suffix
                    _logger.info(
                        '{0}: [{1:>4d}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                            log_name, batch_idx,
                            batch_time=batch_time_m,
                            loss=losses_m,
                            top1=top1_m,
                            top5=top5_m)
                    )


                slide_num += 1
                new_slide = True
            # # ROC
            # run.log({"roc_eval": wandb.plot.roc_curve(target.cpu().detach(), output.cpu().detach())})
            # # Precision-Recall
            # run.log({"pr_eval": wandb.plot.pr_curve(target.cpu().detach(), output.cpu().detach())})
            # # Confusion Matrices
            # roc_auc_eval = roc_auc_score(target.cpu().detach(), output.cpu().detach()[:, 1])
            # run.log({"old_auc_eval": roc_auc_eval if roc_auc_eval.size == 1 else roc_auc_eval[0]})
    if not args.extract_features:
        all_targets = all_targets[1:]
        all_targets_slide_level = all_targets_slide_level[1:]

        all_outputs = all_outputs[1:]
        all_outputs_slide_level = all_outputs_slide_level[1:]

        roc_auc_per_slide = roc_auc_score(all_targets_slide_level, all_outputs_slide_level[:, 1])
        auc_per_slide = roc_auc_per_slide if roc_auc_per_slide.size == 1 else roc_auc_per_slide[0]
        
        roc_auc_per_patch = roc_auc_score(all_targets, all_outputs[:, 1])
        auc_per_patch = roc_auc_per_patch if roc_auc_per_patch.size == 1 else roc_auc_per_patch[0]

        metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])
        # run.log({"auc_eval_per_minibatch": auces_per_minibatch_m.avg})
        run.log({"auc_eval_per_batch": auc_per_patch})
        run.log({"auc_eval_per_slide": auc_per_slide})
    
    return metrics
    
    
if __name__ == '__main__':
    main()