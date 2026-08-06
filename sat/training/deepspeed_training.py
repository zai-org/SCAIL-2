# coding=utf-8
# Rewrite by Ming Ding, Tsinghua University
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import math
import numpy as np
import torch
from collections import defaultdict
from datetime import datetime
from contextlib import ExitStack
from packaging import version

import torch.distributed as dist
import deepspeed
try:
    from torch.distributed._tensor import init_device_mesh
    if version.parse(torch.__version__.split('+')[0]) < version.parse('2.6'):
        from torch.distributed._composable.fsdp import fully_shard, CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, register_fsdp_forward_method
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions, set_model_state_dict, set_optimizer_state_dict
    else:
        from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, register_fsdp_forward_method, FullyShardedDataParallel
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions, set_model_state_dict, set_optimizer_state_dict
    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False
import wandb

from .learning_rates import AnnealingLR
from .model_io import load_checkpoint, save_checkpoint

from .utils import Timers
from .utils import report_memory
from .utils import print_args
from .utils import get_sample_writer
from .utils import init_wandb_writer

from sat import mpu
from sat.data_utils import make_loaders
from sat.transformer_defaults import NO_WD_MODULES
from sat.helpers import print_rank0, print_all
from sat.model.base_model import get_model
from sat.training.model_io import get_checkpoint_tracker_filename
try:
    import wandb
except ImportError:
    print("wandb not installed.")
import warnings
import functools

_fsdp2_enabled = False

def clip_grad_norm_fsdp2(
    parameters,
    max_norm: float,
    norm_type: float = 2.0,
    process_group=None,
    cpu_offload: bool = False,
) -> torch.Tensor:
    """Clips gradient norm of an iterable of parameters.

    This is adapted from fsdp clip_grad_norm and also takes into
    account gradient averaging with DDP.
    """
    grads = []
    grads_dtypes = []
    sharded_grads = []
    nonsharded_grads = []

    for p in parameters:
        grad = p.grad
        if grad is None:
            continue
        grads_dtypes.append(grad.dtype)
        if hasattr(grad, "to_local"):
            grad_local = grad.to_local()
            sharded_grads.append(grad_local.view(-1).float())
        else:
            nonsharded_grads.append(grad.view(-1).float())
        grads.append(grad)

    zero_tensor = torch.tensor(0.0, device=grads[0].device) if grads else torch.tensor(0.0)
    if norm_type == math.inf:
        local_sharded_norm = (
            torch.cat(sharded_grads).abs().max() if sharded_grads else zero_tensor
        )
        local_nonsharded_norm = (
            torch.cat(nonsharded_grads).abs().max() if nonsharded_grads else zero_tensor
        )
    else:
        local_sharded_norm = (
            torch.norm(torch.cat(sharded_grads), norm_type) if sharded_grads else zero_tensor
        )
        local_nonsharded_norm = (
            torch.norm(torch.cat(nonsharded_grads), norm_type) if nonsharded_grads else zero_tensor
        )

    pg = process_group if process_group is not None else dist.group.WORLD
    if norm_type == math.inf:
        total_norm = local_sharded_norm.clone()
        dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pg)
        total_norm = torch.maximum(total_norm, local_nonsharded_norm)
    else:
        sharded_sq = local_sharded_norm.pow(norm_type)
        dist.all_reduce(sharded_sq, op=dist.ReduceOp.SUM, group=pg)
        total_norm = sharded_sq
        if nonsharded_grads:
            total_norm += local_nonsharded_norm.pow(norm_type)
        total_norm = total_norm ** (1.0 / norm_type)

    if cpu_offload:
        total_norm = total_norm.cpu()

    # clip
    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = min(clip_coef, 1.0)
    for grad in grads:
        if hasattr(grad, "to_local"):
            grad_local = grad.to_local()
            grad_local.mul_(clip_coef_clamped)
        else:
            grad.mul_(clip_coef_clamped)

    # Promote dtype
    if len(grads_dtypes) == 0:
        warnings.warn("Called distributed FSDP2 clip_grad_norm with no gradients!")
        return total_norm
    total_norm_dtype = functools.reduce(torch.promote_types, grads_dtypes)
    return total_norm.to(total_norm_dtype)

def training_main(args, model_cls, forward_step_function, create_dataset_function, handle_metrics_function=None, init_function=None, collate_fn=None, forward_step_eval=None):
    """Main training program."""
    hooks = {
        'forward_step': forward_step_function,
        'init_function': init_function,
        'create_dataset_function': create_dataset_function,
        'handle_metrics': handle_metrics_function,
        'forward_step_eval': forward_step_eval or forward_step_function
    }

    timers = Timers()  # Timer.

    # Experiment Name
    if args.load and args.mode == 'pretrain':  # continue training
        args.experiment_name = os.path.basename(os.path.normpath(args.load))
    else:
        args.experiment_name = args.experiment_name + '-' +datetime.now().strftime("%m-%d-%H-%M")

    # Pytorch distributed. must before seed. ALREADY MOVED TO arguments.py!
    # if isinstance(model_cls, type):
    #     initialize_distributed(args)
    #     set_random_seed(args.seed)  # Random seeds for reproducability.

    # Data stuff.
    train_data, val_data, test_data = make_loaders(args, hooks['create_dataset_function'], collate_fn=collate_fn)
    if args.epochs:
        args.train_iters = len(train_data)
        if args.eval_interval is None:
            args.eval_interval = len(train_data)//args.epochs
        if args.save_interval is None:
            args.save_interval = len(train_data)//args.epochs

    # Build model
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls
        # for given model, make sure all the params are in the correct device, or the sync param will raise error
        correct_device = torch.device(args.device)
        for param in model.parameters():
            if param.device != correct_device:
                param.data = param.data.to(correct_device)
        # register buffer
        for name, buffer in model.named_buffers():
            if buffer.device != correct_device:
                buffer.data = buffer.data.to(correct_device)
    if args.fsdp2:
        global _fsdp2_enabled
        _fsdp2_enabled = True

    # Config model IO
    if args.resume_save:
        tracker_filename = get_checkpoint_tracker_filename(args.resume_save)
        retry_iteration = 0
        if os.path.isfile(tracker_filename):
            with open(tracker_filename, 'r') as f:
                metastring = f.read().strip()
                try:
                    retry_iteration = int(metastring)
                    # assert os.path.isdir(os.path.join(args.resume_save, f"{retry_iteration}")), f"Checkpoint dir of iteration {retry_iteration} does not exist."
                except ValueError:
                    retry_iteration = 0
        if retry_iteration > 0:
            args.load = args.resume_save    # overload load path if resume_save is set and there is valid checkpoint in the path

    if args.load is not None and not args.fsdp2:
        # FSDP2 loads after fully_shard is applied in setup_model_untrainable_params_and_optimizer.
        args.iteration = load_checkpoint(model, args)
        # if we don't load optim_states, filelock is no more needed.
        # with FileLock("/root/checkpoint_lock", timeout=-1):
        #     args.iteration = load_checkpoint(model, optimizer, args)
    else:
        args.iteration = 0

    if args.resume_save:
        args.save = args.resume_save        # overload save path if resume_save is set
    elif args.save:
        args.save = os.path.join(args.save, args.experiment_name)
    if torch.distributed.get_rank() == 0 and mpu.get_data_parallel_world_size() > 8:
        import fnmatch
        def statistic_num(directory):
            pattern = 'training_config*'
            count = 0
            for filename in os.listdir(directory):
                if fnmatch.fnmatch(filename, pattern):
                    count += 1
            return count
        if os.path.exists(args.save):
            config_num = statistic_num(args.save)
            config_save_path = os.path.join(args.save, f'training_config_{config_num}.yaml' if config_num > 0 else 'training_config.yaml')
        else:
            config_save_path = os.path.join(args.save, 'training_config.yaml')
        from omegaconf import OmegaConf
        configs = [OmegaConf.load(cfg) for cfg in args.base]
        config = OmegaConf.merge(*configs)
        os.makedirs(args.save, exist_ok=True)
        OmegaConf.save(config=config, f=config_save_path)
    torch.distributed.barrier()

    # init hook before building deepspeed model and optimizer
    if hooks['init_function'] is not None:
        hooks['init_function'](args, model)

    # training
    iteration = 0
    if args.train_iters > 0:
        # Optimization related things
        model, optimizer, iteration = setup_model_untrainable_params_and_optimizer(args, model)
        if iteration > 0:
            args.iteration = iteration

        # initialize lr scheduler
        lr_scheduler = get_learning_rate_scheduler(optimizer, args.iteration, args)
        assert isinstance(lr_scheduler, AnnealingLR), \
            'must be sat AnnealingLR, or the lr in param_groups will be wrong.'

        if args.save_at_start:
            save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

        summary_writer = None
        if torch.distributed.get_rank() == 0:
            if args.mode == 'pretrain':
                print_rank0('Pretraining or Continuing training the Model...')
            elif args.mode == 'finetune':
                print_rank0('Finetuning Model...')
            print_args(args)
            summary_writer = get_sample_writer(base=args.summary_dir, name=args.experiment_name, iteration=args.iteration)
            if args.wandb:
                init_wandb_writer(args)
        # Resume data loader if necessary.
        if args.resume_dataloader:
            if not args.iterable_dataset:
                if train_data is not None:
                    train_data.batch_sampler.start_iter = args.iteration % len(train_data)
                if val_data is not None:
                    start_iter_val = (args.train_iters // args.save_interval) * args.eval_interval
                    val_data.batch_sampler.start_iter = start_iter_val % len(val_data)
            else:
                print_rank0('Warning: we cannot resume iterable dataloader. skipping...')

        if args.do_train:
            with ExitStack() as stack:
                def save_on_exit(args_, model_, optimizer_, lr_scheduler_):
                    save_checkpoint(args_.iteration, model_, optimizer_, lr_scheduler_, args_)

                # re-sync random seed, or tensor parallel might be broken (dropout, droppath)
                # TODO add rng states for data parallel and wrap drops in main path.
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed(args.seed)
                # ---------

                iteration, skipped = train(model, optimizer,
                    lr_scheduler,
                    train_data,
                    val_data,
                    timers, args, summary_writer=summary_writer,
                    hooks=hooks
                    )

    # final save
    if args.save and iteration != 0:
        save_checkpoint(iteration, model, optimizer, lr_scheduler, args)

    # final testing
    if args.do_test and test_data is not None:
        prefix = 'the end of training for test data'
        test_loss = evaluate_and_print_results(prefix, iter(test_data),
            model, len(test_data) if args.strict_eval else args.eval_iters, args, timers, True, split='test', hooks=hooks)

    return model


def sync_params_across_ranks(args, model): # TODO: check param diff
    # zero3 don't need to sync
    from sat.helpers import check_if_zero3
    import time
    start = time.time()

    if not check_if_zero3(args): # MARK: assuming params synced across dp
        print_rank0('Syncing parameters...')

        for param in model.parameters():
            if not param.requires_grad or getattr(param, 'lr_scale', 1) == 0:
                continue
            if not param.model_parallel or mpu.get_model_parallel_world_size() == 1:
                # We already keep the same random seed for different ranks
                # However, it is not reliable. Non-model-parallel parameters could be different when initialization.
                dist.broadcast(
                    param.data,
                    src=0, # group is default group
                )
            else:
                dist.broadcast(
                    param.data,
                    src=mpu.get_sequence_data_parallel_src_rank(), # 0 -- mp_size-1
                    group=mpu.get_sequence_data_parallel_group() # src, src + mp_size, src + mp_size * 2, ...
                )
        print_rank0('Finished syncing parameters. %.3fs elapsed.' % (time.time() - start))

def check_param_sync(args, model):
    from sat.helpers import check_if_zero3
    import time
    start = time.time()
    flag = True

    if not check_if_zero3(args): # MARK: assuming params synced across dp
        print_rank0('Checking parameter sync...')

        for name, param in model.named_parameters():
            if not param.requires_grad or getattr(param, 'lr_scale', 1) == 0:
                continue
            if not param.model_parallel or mpu.get_model_parallel_world_size() == 1:
                # We already keep the same random seed for different ranks
                # However, it is not reliable. Non-model-parallel parameters could be different when initialization.
                if dist.get_rank() == 0:
                    gather_list = [torch.zeros_like(param.data) for i in range(dist.get_world_size())]
                else:
                    gather_list = None

                dist.gather(param.data, gather_list, dst=0)
                local_gather_rank = 0
                rank_step = 1
            else:
                if mpu.get_sequence_data_parallel_rank() == 0:
                    gather_list = [torch.zeros_like(param.data) for i in range(mpu.get_sequence_data_parallel_world_size())]
                else:
                    gather_list = None

                dist.gather(param.data, gather_list, dst=mpu.get_sequence_data_parallel_src_rank(), group=mpu.get_sequence_data_parallel_group())
                local_gather_rank = mpu.get_sequence_data_parallel_src_rank()
                rank_step = mpu.get_model_parallel_world_size()

            # check sync
            if dist.get_rank() == local_gather_rank:
                local_rank = dist.get_rank()
                for i, t in enumerate(gather_list):
                    remote_rank = local_rank + i * rank_step
                    diff = (t - param.data).abs().sum()
                    if diff > 0.:
                        print('Diff detected between rank %d and %d at step %d on param:' % (local_rank, remote_rank, args.iteration), name, diff)
                        flag = False

        dist.barrier()
        if flag:
            print('Param sync check passed on rank %d at step %d.' % (dist.get_rank(), args.iteration))
        else:
            print('Param sync check not passed on rank %d at step %d.' % (dist.get_rank(), args.iteration))

        print_rank0('Finished parameter sync check. %.3fs elapsed.' % (time.time() - start))

    return flag

def setup_model_untrainable_params_and_optimizer(args, model, config_params=None):
    """Setup model and optimizer."""
    iteration = 0 #setting for fsdp2 training states reset
    if hasattr(model, 'disable_untrainable_params'):
        model.disable_untrainable_params() # mark trainable params

    if args.train_data is not None:
        if args.deepspeed:
            param_groups = get_optimizer_param_groups(model)
            # sync initialized parameters
            sync_params_across_ranks(args, model)
            print_rank0("DeepSpeed is enabled.", level='DEBUG')
            # checking optimizer
            optimizer_name = args.deepspeed_config.get('optimizer',{}).get('type', '')
            if optimizer_name.startswith('sat.'):
                from importlib import import_module
                from functools import partial
                # split and import
                optimizer_callable = getattr(import_module(optimizer_name.rsplit('.', maxsplit=1)[0]), optimizer_name.split('.')[-1])
                optimizer_callable = partial(optimizer_callable, **args.deepspeed_config.get('optimizer', {}).get('params', {}))
                print_rank0(f'Using optimizer {optimizer_name} from sat.')
                del args.deepspeed_config['optimizer']
            else:
                optimizer_callable = None

            model, optimizer, _, _ = deepspeed.initialize(
                model=model,
                model_parameters=param_groups,
                optimizer=optimizer_callable,
                args=args,
                mpu=mpu,
                dist_init_required=False,
                config_params=args.deepspeed_config
                    if version.parse(deepspeed.version) < version.parse("0.9.0")
                    else None
            )
        elif hasattr(args, 'fsdp2') and args.fsdp2:
            if not HAVE_FSDP2:
                raise ImportError("FSDP2 is not available. Please upgrade to PyTorch >= 2.5")
            print_rank0("FSDP2 is enabled.", level='DEBUG')
            # Setup FSDP2 configuration
            fsdp_config = getattr(args, 'fsdp2_config', {})
            # Mixed precision policy
            mp_policy = None
            if fsdp_config.get('mixed_precision', False):
                param_dtype = getattr(torch, fsdp_config.get('param_dtype', 'float32'))
                reduce_dtype = getattr(torch, fsdp_config.get('reduce_dtype', 'float32'))
                mp_policy = MixedPrecisionPolicy(
                    param_dtype=param_dtype,
                    reduce_dtype=reduce_dtype,
                )
            # print_rank0(f"Using mp_policy {mp_policy}")
            offload_policy = OffloadPolicy()
            if fsdp_config.get('offload_params', False):
                offload_policy = CPUOffloadPolicy(pin_memory=True)
            # Apply fully_shard to submodules first if specified
            if fsdp_config.get('auto_wrap', True):
                import torch.nn as nn
                def is_container(module):
                    return isinstance(module, (nn.ModuleList, nn.ModuleDict, nn.Sequential, nn.ParameterList))
                def recursive_fully_shard(module, prefix, wrap_patterns, min_params, fully_shard, mp_policy, offload_policy, fsdp_config, is_container):
                    for name, child in module.named_children():
                        child_prefix = f"{prefix}.{name}" if prefix else name
                        recursive_fully_shard(
                            child, child_prefix, wrap_patterns, min_params, fully_shard,
                            mp_policy, offload_policy, fsdp_config, is_container
                        )
                    should_wrap = any(pattern in prefix.lower() for pattern in wrap_patterns)
                    if is_container(module):
                        return
                    num_params = sum(p.numel() for p in module.parameters(recurse=False))
                    if should_wrap and num_params >= min_params:
                        fully_shard(
                            module,
                            # mesh=fsdp_mesh,
                            mp_policy=mp_policy,
                            offload_policy=offload_policy,
                            reshard_after_forward=fsdp_config.get('reshard_after_forward', True)
                        )
                        # print_rank0(f"FSDP2: Wrapped module {prefix} with {num_params / 1e6:.2f}M parameters")
                wrap_patterns = fsdp_config.get('wrap_patterns', ['block', 'layer', 'transformer'])
                min_params = fsdp_config.get('min_params_to_wrap', 1e6)
                recursive_fully_shard(
                    model, '', wrap_patterns, min_params, fully_shard,
                    mp_policy, offload_policy, fsdp_config, is_container
                )
             # Wrap the entire model
            model.model.diffusion_model = fully_shard(
                model.model.diffusion_model ,
                # mesh=fsdp_mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                reshard_after_forward=fsdp_config.get('reshard_after_forward', True)
            )
            model.conditioner = fully_shard(
                model.conditioner,
                # mesh=fsdp_mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                reshard_after_forward=fsdp_config.get('reshard_after_forward', True)
            )
            # Setup optimizer for FSDP2
            optimizer_class_name = fsdp_config.get('optimizer', 'AdamW')
            optimizer_params = fsdp_config.get('optimizer_params')
            optimizer = torch.optim.AdamW(
                # [param for param in model.parameters() if param.requires_grad],
                get_optimizer_param_groups(model),
                lr=optimizer_params['lr'],
                betas=(optimizer_params['betas'][0], optimizer_params['betas'][1]),
                weight_decay=optimizer_params['weight_decay']
            )

            if args.load is not None:
                load_path = args.load
                from sat.training.model_io import get_checkpoint_iteration, get_checkpoint_name
                iteration, release, success = get_checkpoint_iteration(load_path)
                checkpoint_name = os.path.join(load_path, str(iteration), f'fsdp2_rank_0000_checkpoint.pt')
                if not os.path.exists(checkpoint_name):
                    # Fallback to regular checkpoint for compatibility
                    checkpoint_name = get_checkpoint_name(load_path, iteration, release)
                    print_rank0(f'FSDP2 checkpoint not found, trying regular checkpoint: {checkpoint_name}')
                sd = torch.load(checkpoint_name, map_location='cpu')
                print_rank0(f'Loading checkpoint {args.load}')
                if hasattr(model, 'module'):
                    module = model.module
                else: # inference without deepspeed or using FSDP2
                    module = model
                set_model_state_dict(
                    model=module,
                    model_state_dict=sd['module'],
                    options=StateDictOptions(
                        full_state_dict=True,
                        broadcast_from_rank0=False, # TODO: upgrade
                    ),
                )
                if args.mode == 'finetune' and not args.resume_save:
                    iteration = 0
                elif args.mode == 'pretrain' and not args.no_load_rng: # rng states.
                    try:
                        random.setstate(sd['random_rng_state'])
                        np.random.set_state(sd['np_rng_state'])
                        torch.set_rng_state(sd['torch_rng_state'])
                        torch.cuda.set_rng_state(sd['cuda_rng_state'])
                        # mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])

                        #set optimizer state dict
                        set_optimizer_state_dict(
                            model=module,
                            optimizers=optimizer,
                            optim_state_dict=sd['optimizer'],
                            options=StateDictOptions(
                                full_state_dict=True,
                                broadcast_from_rank0=False,
                            ),
                            )
                    except KeyError:
                        print_rank0('Unable to load optimizer from checkpoint {}, exiting. '
                                    'Specify --no-load-rng or --finetune to prevent '
                                    'attempting to load the random '
                                    'state.'.format(checkpoint_name))
                        exit()
                elif args.mode == 'inference':
                    module.eval()

                if mpu.get_data_parallel_rank() == 0:
                    print_all('> successfully loaded {}'.format(checkpoint_name))
                del sd
            # sharded parameters are float32
            for param in model.model.diffusion_model.parameters():
                assert param.dtype == torch.float32

            # unsharded parameters are bfloat16
            model.model.diffusion_model.unshard()
            for param in model.model.diffusion_model.parameters(recurse=False):
                assert param.dtype == torch.bfloat16
            model.model.diffusion_model.reshard()

        else:
            raise ValueError('Currently, we only support training with deepspeed.')
    else:
        optimizer = None

    return model, optimizer, iteration


def add_param_by_lr(dic, p, no_weight_decay=False):
    if not hasattr(p, 'lr_scale'):
        dic[None]['params'].append(p)
    else:
        if p.lr_scale not in dic:
            dic[p.lr_scale] = {'params': [], 'lr': p.lr_scale} if not no_weight_decay else {'params': [], 'weight_decay': 0.0, 'lr': p.lr_scale}
        dic[p.lr_scale]['params'].append(p)

def get_params_for_weight_decay_optimization(module):
    weight_decay_params = {None: {'params': [], 'lr': 1.}}
    no_weight_decay_params = {None: {'params': [], 'weight_decay': 0.0, 'lr': 1.}}
    print_rank0(f"{NO_WD_MODULES} is set to no_weight_decay")
    for module_ in module.modules():
        if isinstance(module_, tuple(NO_WD_MODULES)):
            for p in module_._parameters.values():
                if p is not None and p.requires_grad:
                    add_param_by_lr(no_weight_decay_params, p, no_weight_decay=True)
        else:
            for n, p in module_._parameters.items():
                if p is not None and n != 'bias' and p.requires_grad:
                    flag = True if hasattr(p, 'no_weight_decay') and p.no_weight_decay else False
                    if flag:
                        print_rank0(f"{n} is set to no_weight_decay")
                        add_param_by_lr(no_weight_decay_params, p, no_weight_decay=True)
                    else:
                        add_param_by_lr(weight_decay_params, p, no_weight_decay=False)
            for n, p in module_._parameters.items():
                if p is not None and n == 'bias' and p.requires_grad:
                    add_param_by_lr(no_weight_decay_params, p, no_weight_decay=True)
    ret = []
    for v in weight_decay_params.values():
        if len(v['params']) != 0:
            ret.append(v)
    for v in no_weight_decay_params.values():
        if len(v['params']) != 0:
            ret.append(v)
    return ret


def get_optimizer_param_groups(model):
    # Build parameter groups (weight decay and non-decay).
    if hasattr(model, 'module'):
        model = model.module
    param_groups = get_params_for_weight_decay_optimization(model)
    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel') and not hasattr(param, 'tensor_model_parallel'):
                param.model_parallel = False
                param.tensor_model_parallel = False
            else:
                assert hasattr(param, 'model_parallel') and hasattr(param, 'tensor_model_parallel'), "model_parallel and tensor_model_parallel should both be set or unset!"
    return param_groups


def get_learning_rate_scheduler(optimizer, iteration, args,
                                auto_warmup_steps=0, auto_warmup_rate=0.05):
    """Build the learning rate scheduler."""

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = max(iteration - auto_warmup_steps, 0)
    if args.mode == 'pretrain' and iteration == 0:
        auto_warmup_steps = 0
    # If init_step <= current_steps <= init_step + auto_warmup_steps,
    # lr = auto_warmup_rate * args.lr.
    # This overrides other rules.
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               decay_ratio=args.lr_decay_ratio,
                               auto_warmup_steps=auto_warmup_steps,
                               auto_warmup_rate=auto_warmup_rate
                               )

    return lr_scheduler


def train(model, optimizer, lr_scheduler,
        train_data, val_data, timers, args,
        summary_writer=None, hooks={}):
    """Train the model."""
    if train_data is not None:
        train_data_iterator = iter(train_data)
    else:
        train_data_iterator = None
    if val_data is not None:
        val_data_iterator = iter(val_data)
    else:
        val_data_iterator = None

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_lm_loss = 0.0
    total_metrics = defaultdict(float)
    total_metrics_cnt = defaultdict(int)

    # Iterations.
    skipped_iters = 0

    timers('interval time').start()
    report_memory_flag = True
    while args.iteration < args.train_iters:
        if args.profiling != -1 and args.iteration == args.profiling:
            torch.cuda.cudart().cudaProfilerStart()

        if args.profiling != -1 and args.iteration >= args.profiling:
            torch.cuda.nvtx.range_push("iteration{}".format(args.iteration))
        lm_loss, skipped_iter, metrics, additional_log_dict = train_step(train_data_iterator,
                                                    model,
                                                    optimizer,
                                                    lr_scheduler,
                                                    args, timers, hooks=hooks)
        skipped_iters += skipped_iter
        if args.profiling != -1 and args.iteration >= args.profiling:
            torch.cuda.nvtx.range_pop()
        args.iteration += 1
        # Update losses.
        total_lm_loss += lm_loss.data.detach().float()
        for name in metrics:
            if not 'eval' in name:
                assert len(metrics[name].shape)==0, 'metrics without eval must be scalar'
                value = metrics[name].data.detach().float().item()
                if value > -99:
                    total_metrics[name] += value
                    total_metrics_cnt[name] += 1

        # Logging.
        if args.iteration % args.log_interval == 0:
            learning_rate = optimizer.param_groups[0]['lr']
            avg_lm_loss = total_lm_loss.item() / args.log_interval
            # average img & txt loss
            avg_metrics = {}
            for key in total_metrics:
                avg_metrics[key] = total_metrics[key] / total_metrics_cnt[key] # args.log_interval

            elapsed_time = timers('interval time').elapsed()
            report_iteration_metrics(summary_writer, optimizer, learning_rate, avg_lm_loss,
                                     elapsed_time * 1000.0 / args.log_interval, args.iteration, args.train_iters, args,
                                     avg_metrics, additional_log_dict)
            total_lm_loss = 0.0
            total_metrics = defaultdict(float)
            total_metrics_cnt = defaultdict(int)
            if report_memory_flag:
                report_memory('after {} iterations'.format(args.iteration))
                report_memory_flag = False

            timers.log(['forward', 'backward', 'allreduce', 'optimizer',
                        'batch generator', 'data loader'],
                       normalizer=args.log_interval)
        # Checkpointing
        if args.save and args.save_interval and args.iteration % args.save_interval == 0:
            save_checkpoint(args.iteration, model, optimizer, lr_scheduler, args)

        # Evaluation
        if args.eval_interval and args.iteration % args.eval_interval == 0 and args.do_valid:
            if args.strict_eval:
                val_data_iterator = iter(val_data)
                eval_iters = len(val_data)
            else:
                eval_iters = args.eval_iters
            prefix = 'iteration {}'.format(args.iteration)
            evaluate_and_print_results(
                prefix, val_data_iterator, model, eval_iters, args, timers, False, step=args.iteration, split='val', summary_writer=summary_writer, hooks=hooks)

        if args.empty_cache_interval > 0 and args.iteration % args.empty_cache_interval == 0:
            torch.cuda.empty_cache()

        if args.param_sync_check_interval > 0 and args.iteration % args.param_sync_check_interval == 0:
            check_param_sync(args, model)

        if args.force_param_sync_interval > 0 and args.iteration % args.force_param_sync_interval == 0:
            sync_params_across_ranks(args, model)

        if args.exit_interval and args.iteration % args.exit_interval == 0:
            torch.distributed.barrier()
            time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            rank = torch.distributed.get_rank()
            print_all('rank: {} | time: {} | exiting the program at iteration {}'.
                  format(rank, time_str, args.iteration), flush=True)
            exit()
    if args.profiling != -1:
        torch.cuda.cudart().cudaProfilerStop()

    return args.iteration, skipped_iters


def train_step(data_iterator, model, optimizer, lr_scheduler,
               args, timers, hooks=None, single_step=False, **kwargs):
    """Single training step."""
    if hooks is None:
        hooks = {}
    lm_loss_total, metrics_total, count, metrics_count = 0.0, {}, 0, {}
    forward_step = hooks['forward_step']
    grad_accumulation_steps = getattr(args, 'fsdp_gradient_accumulation_steps', 1)
    while True:
        profiling_flag = (args.profiling != -1 and args.iteration >= args.profiling)
        # Forward model for one step.
        if profiling_flag:
            torch.cuda.nvtx.range_push("forward")
        timers('forward').start()
        forward_ret = forward_step(data_iterator, model, args, timers, **kwargs)
        if isinstance(forward_ret, tuple):
            lm_loss, metrics, additional_log_dict = forward_ret
        else:
            lm_loss, metrics, additional_log_dict = forward_ret, {}, {}
        timers('forward').stop()
        if profiling_flag:
            torch.cuda.nvtx.range_pop()

        # Check nan or inf in forward, preventing it from interfering loss scaler,
        # and all reduce metrics by the way
        if profiling_flag:
            torch.cuda.nvtx.range_push("loss_and_metrics")
        lm_loss_reduced = lm_loss.detach().clone()
        torch.distributed.all_reduce(lm_loss_reduced.data)
        lm_loss_reduced.data = lm_loss_reduced.data / args.world_size

        loss_checker = lm_loss_reduced
        for name in metrics:
            if not 'eval' in name:
                metrics[name] = metrics[name].detach().clone()
                if metrics[name].data.item() == -100:
                    cnt = torch.zeros(1, dtype=torch.int64, device=metrics[name].data.device)
                    metrics[name].data = torch.tensor(0., device=metrics[name].data.device)
                else:
                    cnt = torch.ones(1, dtype=torch.int64, device=metrics[name].data.device)
                torch.distributed.all_reduce(metrics[name].data)
                torch.distributed.all_reduce(cnt)
                if cnt.item() == 0:
                    metrics[name].data = torch.tensor(-100, device=metrics[name].data.device)
                else:
                    metrics[name].data /= cnt.cpu().item() # args.world_size
                loss_checker = loss_checker + metrics[name]
        if loss_checker.isnan().any() or loss_checker.isinf().any():
            print_all('Skipping backward and optimizer step for nan or inf in forwarding metrics/loss!')
            return lm_loss.detach(), 1, metrics

        # Accumulate the statistics
        lm_loss_total += lm_loss_reduced
        for name in metrics:
            if name not in metrics_total:
                metrics_total[name] = torch.tensor(0.0, device=metrics[name].data.device)
            if name not in metrics_count:
                metrics_count[name] = 0
            if metrics[name].data.item() != -100:
                metrics_total[name] += metrics[name]
                metrics_count[name] += 1
        count += 1
        if profiling_flag:
            torch.cuda.nvtx.range_pop()

        if profiling_flag:
            torch.cuda.nvtx.range_push("backward")
        # Calculate gradients, reduce across processes, and clip.
        timers('backward').start()
        backward_step(optimizer, model, lm_loss, args, timers)
        timers('backward').stop()
        if profiling_flag:
            torch.cuda.nvtx.range_pop()
        # Update parameters.
        skipped_iter, complete = 0, False
        if profiling_flag:
            torch.cuda.nvtx.range_push("optimizer")
        timers('optimizer').start()
        if args.deepspeed:
            if model.is_gradient_accumulation_boundary():
                model.step()
                complete = True
                if not (args.fp16 and optimizer.overflow):
                    lr_scheduler.step()
                else:
                    skipped_iter = 1
            else:
                model.step()
        elif hasattr(args, 'fsdp2') and args.fsdp2:
            if count % grad_accumulation_steps == 0:
                fsdp_config = getattr(args, 'fsdp2_config', {})
                max_grad_norm = fsdp_config.get('max_grad_norm', 1.0)
                # print(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10000))
                # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                grad_norm = clip_grad_norm_fsdp2(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                complete = True
            else:
                complete = False
        else:
            raise ValueError('Currently, we only support training with deepspeed.')
        timers('optimizer').stop()
        if profiling_flag:
            torch.cuda.nvtx.range_pop()
        if complete or single_step:
            break
    grad_name = 'grad_norm'
    if args.deepspeed:
        additional_log_dict[grad_name] = model.get_global_grad_norm().item()
    elif hasattr(args, 'fsdp2') and args.fsdp2:
        additional_log_dict[grad_name] = grad_norm
    else:
        additional_log_dict[grad_name] = 0.0
    lm_loss_total /= count
    metrics_total = {key: torch.tensor(-100, device=metrics_total[key].data.device) if metrics_count[key] == 0 else value / metrics_count[key] for key, value in metrics_total.items()}
    return lm_loss_total, skipped_iter, metrics_total, additional_log_dict


def backward_step(optimizer, model, loss, args, timers):
    """Backward step."""

    # Backward pass.
    if args.deepspeed:
        model.backward(loss)
    elif hasattr(args, 'fsdp2') and args.fsdp2:
        # FSDP2 backward pass
        loss = loss / getattr(args, 'fsdp_gradient_accumulation_steps', 1)
        loss.backward()
    else:
        raise ValueError('Currently, we only support training with deepspeed or fsdp2.')

    timers('allreduce').reset()

    return

def evaluate(data_iterator, model, eval_iters, args, timers, split, verbose=False, has_last=True, hooks={}):
    """Evaluation."""
    forward_step = hooks['forward_step_eval']
    # Turn on evaluation mode which disables dropout.
    model.eval()
    rank = torch.distributed.get_rank(group=mpu.get_data_parallel_group())
    total_lm_loss, metrics_total = 0, {}
    if split=='val':
        last_shape = args.val_last_shape
        drop_number = args.val_drop_number
    else:
        assert split=='test'
        last_shape = args.test_last_shape
        drop_number = args.test_drop_number
    is_scalar = {}
    with torch.no_grad():
        iteration = 0
        while iteration < eval_iters:
            iteration += 1
            if verbose and iteration % args.log_interval == 0:
                print_rank0('Evaluating iter {}/{}'.format(iteration, eval_iters))
            # Forward evaluation.
            # try:
            lm_loss, metrics = forward_step(data_iterator, model, args, timers)
            '''when contiguous memory optimizations are enabled, the buffers
            allocated by the optimizations are deallocated during backward pass
            in the absence of backward pass the buffers should be reset after each
            forward pass'''
            if args.deepspeed and args.deepspeed_activation_checkpointing:
                deepspeed.checkpointing.reset()
            total_lm_loss += lm_loss.data.detach().float().item()
            is_last = True if iteration == eval_iters and args.strict_eval and len(last_shape)>0 else False
            for name in metrics:
                if name not in metrics_total:
                    metrics_total[name] = []
                is_scalar[name] = True if len(metrics[name].shape)==0 else False
                shape = list(metrics[name].shape)
                if not is_scalar[name] and is_last and metrics[name].shape[0] != last_shape[0]:
                    # pad tensor's first dim to args.batch_size
                    metrics[name] = torch.concat([metrics[name], torch.zeros([last_shape[0]-metrics[name].shape[0]] + shape[1:], dtype=metrics[name].dtype, device=metrics[name].device)])
                if rank==0:
                    metrics_gathered = [torch.zeros_like(metrics[name], dtype=metrics[name].dtype, device=metrics[name].device) for _ in range(args.world_size)]
                else:
                    # metrics_gathered = None
                    metrics_gathered = [torch.zeros_like(metrics[name], dtype=metrics[name].dtype, device=metrics[name].device) for _ in range(args.world_size)]
                # torch.distributed.gather(metrics[name], metrics_gathered, 0)
                torch.distributed.all_gather(metrics_gathered, metrics[name])

                if rank==0:
                    gathered_len = len(metrics_gathered) if not is_last else len(metrics_gathered) - drop_number * args.model_parallel_size
                    for i in range(gathered_len):
                        if is_scalar[name] or not is_last:
                            metrics_total[name].append(metrics_gathered[i].data.cpu())
                        else:
                            metrics_total[name].append(metrics_gathered[i][:last_shape[i]].data.cpu())
    # Move model back to the train mode.
    model.train()

    total_lm_loss /= eval_iters
    # metrics_avg = {key: value / eval_iters for key, value in metrics_total.items()}
    if rank==0:
        for name in metrics_total:
            if is_scalar[name]:
                metrics_total[name] = torch.stack(metrics_total[name], dim=0)
            else:
                metrics_total[name] = torch.concat(metrics_total[name], dim=0)
        if hooks['handle_metrics'] is not None:
            metrics = hooks['handle_metrics'](metrics_total)
        else:
            for name in metrics_total:
                assert is_scalar[name], 'you must return scalar metrics or implement handle_metrics hooks'
            metrics = {key: sum(value.split(1,0))/len(value) for key, value in metrics_total.items()}
    else:
        metrics = None
    return total_lm_loss, metrics

def evaluate_and_print_results(prefix, data_iterator, model, eval_iters,
                            args, timers, has_last, split, verbose=False, step=None, summary_writer=None, hooks={}):
    """Helper function to evaluate and dump results on screen."""
    lm_loss, metrics = evaluate(data_iterator, model, eval_iters, args, timers, split, verbose, has_last, hooks=hooks)
    lm_ppl = math.exp(min(20, lm_loss))
    if torch.distributed.get_rank(group=mpu.get_data_parallel_group())==0:
        report_evaluate_metrics(summary_writer, prefix, lm_loss, lm_ppl, step, args, metrics)
    return lm_loss


def report_iteration_metrics(summary_writer, optimizer, lr, loss, elapsed_time, step, total_step, args, avg_metrics, additional_log_dict):
    log_string = ' iteration {:8d}/{:8d} |'.format(step, total_step)
    log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(elapsed_time)
    log_string += ' learning rate {:.3E} |'.format(lr)
    log_string += ' total loss {:.6E} |'.format(loss)
    for key in avg_metrics:
        log_string += ' {} {:.6E} |'.format(key, avg_metrics[key])
    if args.fp16:
        if args.deepspeed:
            log_string += ' loss scale {:.1f} |'.format(optimizer.cur_scale)
        elif hasattr(args, 'fsdp2') and args.fsdp2:
            log_string += ' loss scale N/A (FSDP2) |'
        else:
            log_string += ' loss scale {:.1f} |'.format(optimizer.loss_scale)
    log_string += 'speed {:.2f} samples/(min*GPU)'.format(
        (args.gradient_accumulation_steps * args.batch_size / args.model_parallel_size / (elapsed_time / 60000.0)))
    print_rank0(log_string)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/lr', lr, step)
        summary_writer.add_scalar(f'Train/train_loss', loss, step)
        summary_writer.add_scalar(f'Train/elapsed_time', elapsed_time, step)
        for key in avg_metrics:
            summary_writer.add_scalar('Train/'+key, avg_metrics[key], step)
    if args.wandb and torch.distributed.get_rank() == 0:
        log_dict = {
            "Train/lr": lr,
            "Train/train_loss": loss,
            "Train/elapsed_time": elapsed_time
            }
        log_dict.update(additional_log_dict)
        for key in avg_metrics:
            log_dict["Train/" + key] = avg_metrics[key]
        wandb.log(log_dict, step=step, commit=True)


def report_evaluate_metrics(summary_writer, prefix, loss, ppl, step, args, avg_metrics):
    string = ' validation loss at {} | '.format(prefix)
    string += 'loss: {:.6E} | '.format(loss)
    string += 'PPL: {:.6E}'.format(ppl)
    for key in avg_metrics:
        string += ' {} {:.6E} |'.format(key, avg_metrics[key].item())
    length = len(string) + 1
    print_rank0('-' * 100)
    print_rank0('-' * length)
    print_rank0(string)
    print_rank0('-' * length)
    if summary_writer is not None:
        summary_writer.add_scalar(f'Train/valid_ppl', ppl, step)
        summary_writer.add_scalar(f'Train/valid_loss', loss, step)
        for key in avg_metrics:
            summary_writer.add_scalar('Train/valid_'+key, avg_metrics[key], step)
    if args.wandb and torch.distributed.get_rank() == 0:
        log_dict = {
            "Train/valid_ppl": ppl,
            "Train/valid_loss": loss,
            }
        for key in avg_metrics:
            log_dict["Train/valid_" + key] = avg_metrics[key]
        wandb.log(log_dict, step=step, commit=True)
