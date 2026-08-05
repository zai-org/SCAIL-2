import math
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Union, Optional
from omegaconf import ListConfig, OmegaConf
from copy import deepcopy
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import copy

from sgm.modules.diffusionmodules.loss import guidance_scale_embedding
from sgm.modules import UNCONDITIONAL_CONFIG
from sgm.modules.autoencoding.temporal_ae import VideoDecoder
from sgm.modules.diffusionmodules.wrappers import OPENAIUNETWRAPPER
from sgm.util import (
    default,
    disabled_train,
    get_obj_from_str,
    instantiate_from_config,
    log_txt_as_img,
    SeededNoise
)
from sgm.modules.diffusionmodules.util import (
    timestep_embedding
)
from sat import mpu
from sat.helpers import print_rank0
from sat.training.model_io import load_checkpoint
from sat.mpu.operation import mp_split_model_rank0, mp_split_model_receive, mp_merge_model_rank0, mp_merge_model_send
from sat.arguments import update_args_with_file, overwrite_args_by_dict, set_random_seed
from sat.mpu.initialize import get_node_rank, get_model_parallel_rank, destroy_model_parallel, initialize_model_parallel
from sat.model.base_model import get_model
import gc
from sat.arguments import reset_random_seed
import random

class SATVideoDiffusionEngine(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()

        model_config = args.model_config
        # model args preprocess
        log_keys = model_config.get('log_keys', None)
        input_key = model_config.get('input_key', 'mp4')
        network_config = model_config.get('network_config', None)
        network_wrapper = model_config.get('network_wrapper', None)
        denoiser_config = model_config.get('denoiser_config', None)
        sampler_config = model_config.get('sampler_config', None)
        conditioner_config = model_config.get('conditioner_config', None)
        i2v_clip_config = model_config.get('i2v_clip_config', None)
        first_stage_config = model_config.get('first_stage_config', None)
        loss_fn_config = model_config.get('loss_fn_config', None)
        scale_factor = model_config.get('scale_factor', 1.0)
        latent_input = model_config.get('latent_input', False)
        use_pose = model_config.get('use_pose', False)
        disable_first_stage_autocast = model_config.get('disable_first_stage_autocast', False)
        no_cond_log = model_config.get('disable_first_stage_autocast', False)
        untrainable_prefixs = model_config.get('untrainable_prefixs', ['first_stage_model', 'conditioner'])
        compile_model = model_config.get('compile_model', False)
        en_and_decode_n_samples_a_time = model_config.get('en_and_decode_n_samples_a_time', None)
        lora_train = model_config.get('lora_train', False)
        self.use_pd = model_config.get('use_pd', False) # progressive distillation
        self.use_i2v_clip = model_config.get('use_i2v_clip', False) # inspired from wanx-i2v
        self.i2v_encode_video = model_config.get('i2v_encode_video', False) # inspired from wanx-i2v

        self.log_keys = log_keys
        self.input_key = input_key
        self.untrainable_prefixs = untrainable_prefixs
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.lora_train = lora_train
        self.noised_image_input = model_config.get('noised_image_input', False)
        self.noised_image_all_concat = model_config.get('noised_image_all_concat', False)
        self.image_cond_dropout = model_config.get('image_cond_dropout', 0.0)
        self.pose_dropout = model_config.get('pose_dropout', 0.0)

        self.use_pose = use_pose

        if args.fp16:
            dtype = torch.float16
            dtype_str = "fp16"
        elif args.bf16:
            dtype = torch.bfloat16
            dtype_str = "bf16"
        else:
            dtype = torch.float32
            dtype_str = "fp32"
        self.dtype = dtype
        self.dtype_str = dtype_str

        network_config['params']['dtype'] = dtype_str
        network_config['params']['use_i2v_clip'] = self.use_i2v_clip
        model = instantiate_from_config(network_config)
        self.model = get_obj_from_str(default(network_wrapper, OPENAIUNETWRAPPER))(
            model, compile_model=compile_model, dtype=dtype
        )

        self.denoiser = instantiate_from_config(denoiser_config)
        self.sampler = (
            instantiate_from_config(sampler_config)
            if sampler_config is not None
            else None
        )
        self.conditioner = instantiate_from_config(
            default(conditioner_config, UNCONDITIONAL_CONFIG)
        )
        if self.use_i2v_clip:
            self.i2v_clip = instantiate_from_config(i2v_clip_config) if i2v_clip_config is not None else None

        self._init_first_stage(first_stage_config)

        self.loss_fn = (
            instantiate_from_config(loss_fn_config)
            if loss_fn_config is not None
            else None
        )

        self.latent_input = latent_input
        self.scale_factor = scale_factor
        self.disable_first_stage_autocast = disable_first_stage_autocast
        self.no_cond_log = no_cond_log
        self.device = args.device

        if self.use_pd and (args.mode == 'pretrain' or args.mode == 'finetune'):
            from sat.training.model_io import load_checkpoint
            import copy
            print("############# load teacher model")
            self.teacher_model = copy.deepcopy(self.model)
            # load state_dict into CPU        
            sd = torch.load(self.teacher_path, map_location='cpu')
            # if given `prefix`, load a speficic prefix in the checkpoint, e.g. encoder
            prefix = 'model.'
            new_sd = {'module':{}}
            for k in sd:
                if k != 'module':
                    new_sd[k] = sd[k]
            for k in sd['module']:
                if k.startswith(prefix):
                    new_sd['module'][k[len(prefix):]] = sd['module'][k]
            sd = new_sd
            missing_keys, unexpected_keys = self.teacher_model.load_state_dict(sd['module'], strict=False)
            if len(unexpected_keys) > 0:
                print_rank0(
                    f'Will continue but found unexpected_keys! Check whether you are loading correct checkpoints: {unexpected_keys}.')
            if len(missing_keys) > 0:
                print_rank0(f'Warning: Missing keys for inference: {missing_keys}.')

            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad_(False)

    def disable_untrainable_params(self):
        # untrainable_keywords_crossattn = ["cross_attention.query.weight", "cross_attention.query.bias", "cross_attention.key_value.weight", "cross_attention.key_value.bias", "cross_attention.dense.weight", "cross_attention.dense.bias"]
        # untrainable_keywords_proj = ["final_layer.linear"]
        # untrainable_keywords = untrainable_keywords_crossattn + untrainable_keywords_proj
        untrainable_keywords = []
        total_trainable = 0
        if self.lora_train:
            for n, p in self.named_parameters():
                if p.requires_grad == False:
                    continue
                if 'lora_layer' not in n:
                    p.lr_scale = 0
                else:
                    total_trainable += p.numel()
        else:
            for n, p in self.named_parameters():
                if p.requires_grad == False:
                    continue
                flag = False
                for prefix in self.untrainable_prefixs:
                    if n.startswith(prefix) or prefix == "all":
                        flag = True
                        break
                for untrainable_keyword in untrainable_keywords:
                    if untrainable_keyword in n:
                        print(f"debug: {n} is untrainable")
                        flag = True
                        break

                lora_prefix = ['matrix_A', 'matrix_B']
                for prefix in lora_prefix:
                    if prefix in n:
                        flag = False
                        break

                if flag:
                    p.requires_grad_(False)
                else:
                    total_trainable += p.numel()

        print_rank0("***** Total trainable parameters: " + str(total_trainable) + " *****")

    def reinit(self, parent_model=None):
        # reload the initial params from previous trained modules
        # you can also get access to other mixins through parent_model.get_mixin().
        pass

    def _init_first_stage(self, config):
        model = instantiate_from_config(config)
        if not 'wan_vae' in config['target']:
            model = model.eval()
        model.train = disabled_train
        if not 'wan_vae' in config['target']:
            for param in model.parameters():
                param.requires_grad = False
        else:
            for param in model.model.parameters():
                param.requires_grad = False
        self.first_stage_model = model

    def get_input(self, batch):
        assert self.use_pose, "use_pose must be True when latent_input is True"
        if not self.latent_input:
            raise NotImplementedError("get_input only supports latent_input=True for now")
        if 'smpl_render' not in batch:
            raise NotImplementedError("smpl_render and related inputs are required when latent_input is True for now")

        keys = ['smpl_render', 'smpl_render_aug', 'latent_sam', 'latent_sam_aug',
                'latent_ref_mask', 'ref_frame', 'first_frame', 'pixel_first_frame', self.input_key]
        for k in keys:
            if not isinstance(batch[k], torch.Tensor):
                print(f"[get_input] batch['{k}'] is not a tensor: {type(batch[k])}")
        return tuple(batch[k].to(self.dtype) for k in keys)

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        n_samples = default(self.en_and_decode_n_samples_a_time, z.shape[0])
        n_rounds = math.ceil(z.shape[0] / n_samples)
        all_out = []
        for n in range(n_rounds):
            z_now = z[n * n_samples : (n + 1) * n_samples]
            recons = self.first_stage_model.decode(z_now) # b c t h w
            all_out.append(recons)
        out = torch.cat(all_out, dim=0)
        return out

    @torch.no_grad()
    def encode_first_stage(self, x, batch, force_encode=False):
        if not force_encode and self.latent_input:
            return x * self.scale_factor # already encoded # bcthw

        n_samples = default(self.en_and_decode_n_samples_a_time, x.shape[0])
        n_rounds = math.ceil(x.shape[0] / n_samples)
        all_out = []

        for n in range(n_rounds):
            x_now = x[n * n_samples: (n + 1) * n_samples]
            latents = self.first_stage_model.encode(x_now) # b c t h w
            all_out.append(latents)
        z = torch.cat(all_out, dim=0)
        z = self.scale_factor * z # b c t h w
        torch.distributed.broadcast(z, src=mpu.get_data_broadcast_src_rank(), group=mpu.get_data_broadcast_group())
        return z

    def forward(self, x, batch):
        if self.use_pd:
            loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch, self.teacher_model, self.sampler)
        else:
            loss = self.loss_fn(self.model, self.denoiser, self.conditioner, x, batch)
        loss_mean = loss.mean()
        loss_dict = {"diffusion loss": loss_mean}
        return loss_mean, loss_dict

    def add_noise_to_first_frame(self, image):
        sigma = torch.normal(mean=-2.5, std=0.5, size=(image.shape[0],)).to(self.device)
        sigma = torch.exp(sigma).to(image.dtype)
        image_noise = torch.randn_like(image) * sigma[:, None, None, None, None]
        image = image + image_noise
        return image

    def shared_step(self, batch: Dict) -> Any:
        assert self.use_pose, "use_pose must be True when latent_input is True"
        if self.latent_input:
            assert self.i2v_encode_video, "i2v_encode_video must be True when latent_input is True"
            smpl_render, smpl_render_aug, latent_sam, latent_sam_aug, latent_ref_mask, ref_frame_latent, first_frame_latent, pixel_first_frame, x = self.get_input(batch)

            if smpl_render is not None:
                e2e_flag = batch.get('e2e_flag', False)
                for idx in range(smpl_render.shape[0]):
                    e2e_flag_i = e2e_flag[idx].item() if isinstance(e2e_flag, torch.Tensor) else e2e_flag
                    aug_prob = 0.56 if e2e_flag_i else 0.75    # preview: 0.8
                    if random.random() < aug_prob:
                        smpl_render[idx] = smpl_render_aug[idx]
                        latent_sam[idx] = latent_sam_aug[idx]
                smpl_render = smpl_render.permute(0, 2, 1, 3, 4).contiguous()
                latent_sam = latent_sam.permute(0, 2, 1, 3, 4).contiguous()
                latent_ref_mask = latent_ref_mask.permute(0, 2, 1, 3, 4).contiguous()

                if self.pose_dropout > 0:
                    null_smpl = torch.load(f"latents/zero_pose_latent_{smpl_render.shape[1]}_{smpl_render.shape[3]}_{smpl_render.shape[4]}.pt").to(self.device).to(self.dtype)
                else:
                    null_smpl = None
                null_noisy_mask = torch.zeros(smpl_render.shape[1], latent_ref_mask.shape[2], latent_ref_mask.shape[3], latent_ref_mask.shape[4], device=self.device, dtype=self.dtype)  # (T_smpl, C=28, H, W)
            x = x.permute(0, 2, 1, 3, 4).contiguous()
            first_frame_latent = first_frame_latent.permute(0, 2, 1, 3, 4).contiguous()
            ori_image = pixel_first_frame.permute(0, 2, 1, 3, 4).contiguous()    # B C 1 H W -> B 1 C H W
            ref_frame_latent = ref_frame_latent.permute(0, 2, 1, 3, 4).contiguous()


            history_mask = torch.zeros_like(first_frame_latent[:, :, :4, :, :], dtype=torch.bool)   # b t 4 h w
            can_history = batch.get('can_history', None)
            for idx in range(history_mask.shape[0]):
                if can_history is None:
                    flag_i = False
                else:
                    flag_i = can_history[idx].item() if isinstance(can_history, torch.Tensor) else can_history[idx]
                    flag_i = bool(flag_i)
                history_prob = 0.5 if flag_i else 0.15
                if random.random() < history_prob:
                    history_mask[idx, :2] = 1
            batch["history_mask"] = history_mask.to(self.dtype)

            if smpl_render is not None:
                # concat null_noisy_mask along T dim for the whole batch at once
                null_noisy_mask_batch = null_noisy_mask.unsqueeze(0).expand(smpl_render.shape[0], -1, -1, -1, -1)
                latent_ref_mask = torch.cat([latent_ref_mask, null_noisy_mask_batch], dim=1)  # B T C H W
                if self.pose_dropout > 0:
                    ref_mask_flags = batch.get('ref_mask_flag', None)
                    for idx in range(smpl_render.shape[0]):
                        flag_i = ref_mask_flags[idx] if ref_mask_flags is not None else True
                        if isinstance(flag_i, torch.Tensor):
                            flag_i = flag_i.item()
                        if str(flag_i) == 'True' and random.random() < self.pose_dropout:
                            smpl_render[idx] = null_smpl
                batch["concat_smpl_render"] = smpl_render.to(self.dtype)
                batch["concat_latent_sam"] = latent_sam.to(self.dtype)
                batch["concat_latent_ref_mask"] = latent_ref_mask.to(self.dtype)
            batch["concat_images"] = first_frame_latent.to(self.dtype)
            batch["ref_concat"] = ref_frame_latent.to(self.dtype)
            # 这里如果要支持 ref_concat 需要改下 vae encode流程
        else:
            raise NotImplementedError("non-latent input is not implemented yet")

           
        if self.use_i2v_clip:
            image_clip_features = self.i2v_clip.visual(ori_image.permute(0, 2, 1, 3, 4)) # btchw -> bcthw
            batch["image_clip_features"] = image_clip_features.to(self.dtype)

        loss, loss_dict = self(x, batch)
        return loss, loss_dict

    @torch.no_grad()
    def sample(
            self,
            cond: Dict,
            uc: Union[Dict, None] = None,
            batch_size: int = 16,
            shape: Union[None, Tuple, List] = None,
            prefix = None,
            concat_images = None,
            ofs = None,
            fps = None,
            tile_indices = None,
            **kwargs,
    ):
        randn = torch.randn(batch_size, *shape).to(torch.float32).to(self.device)

        if hasattr(self, "seeded_noise"):
            randn = self.seeded_noise(randn)
        if hasattr(self.loss_fn, "block_scale") and self.loss_fn.block_scale is not None:
            randn = self.loss_fn.get_blk_noise(randn)

        if prefix is not None:
            randn = torch.cat([prefix, randn[:, prefix.shape[1]:]], dim=1)

        #broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        sp_size = mpu.get_sequence_parallel_world_size()
        if mp_size > 1 or sp_size > 1:
            torch.distributed.broadcast(randn, src=mpu.get_data_broadcast_src_rank(), group=mpu.get_data_broadcast_group())

        chunk_dim = None
        if sp_size > 1:
            sp_rank = mpu.get_sequence_parallel_rank()
            h, w = shape[-2:]
            if h < w:
                chunk_dim = 3
            else:
                chunk_dim = 4
            randn = torch.chunk(randn, sp_size, dim=chunk_dim)[sp_rank]
            if "concat" in cond.keys():
                uc['concat'] = torch.chunk(uc['concat'], sp_size, dim=chunk_dim)[sp_rank]
                cond['concat'] = torch.chunk(cond['concat'], sp_size, dim=chunk_dim)[sp_rank]
            if "concat_images" in cond.keys():
                uc['concat_images'] = torch.chunk(uc['concat_images'], sp_size, dim=chunk_dim)[sp_rank]
                cond['concat_images'] = torch.chunk(cond['concat_images'], sp_size, dim=chunk_dim)[sp_rank]
            if "smpl_tiled" in cond.keys():
                uc['smpl_tiled'] = torch.chunk(uc['smpl_tiled'], sp_size, dim=chunk_dim+1)[sp_rank]
                cond['smpl_tiled'] = torch.chunk(cond['smpl_tiled'], sp_size, dim=chunk_dim+1)[sp_rank]
            if "ref_concat" in cond.keys():
                uc['ref_concat'] = torch.chunk(uc['ref_concat'], sp_size, dim=chunk_dim)[sp_rank]
                cond['ref_concat'] = torch.chunk(cond['ref_concat'], sp_size, dim=chunk_dim)[sp_rank]
            if "concat_pose" in cond.keys():
                uc['concat_pose'] = torch.chunk(uc['concat_pose'], sp_size, dim=chunk_dim)[sp_rank]
                cond['concat_pose'] = torch.chunk(cond['concat_pose'], sp_size, dim=chunk_dim)[sp_rank]
            if "concat_smpl_render" in cond.keys():
                uc['concat_smpl_render'] = torch.chunk(uc['concat_smpl_render'], sp_size, dim=chunk_dim)[sp_rank]
                cond['concat_smpl_render'] = torch.chunk(cond['concat_smpl_render'], sp_size, dim=chunk_dim)[sp_rank]
            if "concat_latent_sam" in cond.keys():
                uc['concat_latent_sam'] = torch.chunk(uc['concat_latent_sam'], sp_size, dim=chunk_dim)[sp_rank]
                cond['concat_latent_sam'] = torch.chunk(cond['concat_latent_sam'], sp_size, dim=chunk_dim)[sp_rank]
            if "concat_latent_ref_mask" in cond.keys():
                uc['concat_latent_ref_mask'] = torch.chunk(uc['concat_latent_ref_mask'], sp_size, dim=chunk_dim)[sp_rank]
                cond['concat_latent_ref_mask'] = torch.chunk(cond['concat_latent_ref_mask'], sp_size, dim=chunk_dim)[sp_rank]
            if "concat_cheek_hands" in cond.keys():
                uc['concat_cheek_hands'] = torch.chunk(uc['concat_cheek_hands'], sp_size, dim=chunk_dim)[sp_rank]
                cond['concat_cheek_hands'] = torch.chunk(cond['concat_cheek_hands'], sp_size, dim=chunk_dim)[sp_rank]
                # smpl_tiled 前面多一维N
            if "history" in cond.keys():
                uc['history'] = torch.chunk(uc['history'], sp_size, dim=chunk_dim)[sp_rank]
                cond['history'] = torch.chunk(cond['history'], sp_size, dim=chunk_dim)[sp_rank]
            if "history_mask" in cond.keys():
                uc['history_mask'] = torch.chunk(uc['history_mask'], sp_size, dim=chunk_dim)[sp_rank]
                cond['history_mask'] = torch.chunk(cond['history_mask'], sp_size, dim=chunk_dim)[sp_rank]
        # 这里拉取新版commit后去掉了pd逻辑
        denoiser = lambda input, sigma, c, **addtional_model_inputs: self.denoiser(
            self.model, input, sigma, c, concat_images=concat_images, chunk_dim=chunk_dim, **addtional_model_inputs
        )
        if tile_indices is not None:
            samples = self.sampler(denoiser, randn, cond, uc=uc, tile_indices=tile_indices)
        else:
            samples = self.sampler(denoiser, randn, cond, uc=uc)
        samples = samples.to(self.dtype)
        if sp_size > 1:
            sp_rank = mpu.get_sequence_parallel_rank()
            gather_list = [torch.zeros_like(samples) for _ in range(sp_size)] if sp_rank == 0 else None
            torch.distributed.gather(samples, dst=mpu.get_sequence_parallel_src_rank(), gather_list=gather_list, group=mpu.get_sequence_parallel_group())
            if sp_rank == 0:
                samples = torch.concat(gather_list, dim=chunk_dim)

        return samples
        


    # @torch.no_grad()
    # def sample_with_pose_cond(
    #         self,
    #         c1_2: Dict,
    #         c1: Dict,
    #         uc: Union[Dict, None] = None,
    #         batch_size: int = 16,
    #         shape: Union[None, Tuple, List] = None,
    #         prefix = None,
    #         concat_images = None,
    #         ofs = None,
    #         **kwargs,
    # ):
    #     randn = torch.randn(batch_size, *shape).to(torch.float32).to(self.device)
    #     #debug !!!!!!!
    #     # breakpoint()
    #     # randn = torch.ones_like(randn)
    #     # randn = torch.load('randn.pt').to(self.device)

    #     if hasattr(self, "seeded_noise"):
    #         randn = self.seeded_noise(randn)
    #     if hasattr(self.loss_fn, "block_scale") and self.loss_fn.block_scale is not None:
    #         randn = self.loss_fn.get_blk_noise(randn)

    #     if prefix is not None:
    #         randn = torch.cat([prefix, randn[:, prefix.shape[1]:]], dim=1)

    #     #broadcast noise
    #     mp_size = mpu.get_model_parallel_world_size()
    #     if mp_size > 1:
    #         global_rank = torch.distributed.get_rank() // mp_size
    #         src = global_rank * mp_size
    #         torch.distributed.broadcast(randn, src=src, group=mpu.get_model_parallel_group())
    #     sp_size = mpu.get_sequence_parallel_world_size()
    #     chunk_dim = None
    #     if sp_size > 1:
    #         src = mpu.get_sequence_parallel_src_rank()
    #         torch.distributed.broadcast(randn, src=src, group=mpu.get_sequence_parallel_group())
    #         local_rank = mpu.get_sequence_parallel_rank()
    #         h, w = shape[-2:]
    #         if h < w:
    #             chunk_dim = 3
    #         else:
    #             chunk_dim = 4
    #         randn = torch.chunk(randn, sp_size, dim=chunk_dim)[local_rank]
    #         if "concat" in c1_2.keys():
    #             c1['concat'] = torch.chunk(c1['concat'], sp_size, dim=chunk_dim)[local_rank]
    #             c1_2['concat'] = torch.chunk(c1_2['concat'], sp_size, dim=chunk_dim)[local_rank]
    #             uc['concat'] = torch.chunk(uc['concat'], sp_size, dim=chunk_dim)[local_rank]

    #     if self.use_pd == True:
    #         scale = 1.0
    #         scale_emb = timestep_embedding(randn.new_ones([batch_size]) * self.sampler.guider.scale, self.model.diffusion_model.cfg_embed_dim).to(self.dtype)
    #     else:
    #         scale = None
    #         scale_emb = None

    #     denoiser = lambda input, sigma, c, **addtional_model_inputs: self.denoiser(
    #         self.model, input, sigma, c, concat_images=concat_images, chunk_dim=chunk_dim, **addtional_model_inputs
    #     )
    #     samples = self.sampler(denoiser, randn, c1_2, c1, uc=uc, scale=scale, scale_emb=scale_emb, ofs=ofs)
    #     samples = samples.to(self.dtype)
    #     if sp_size > 1:
    #         gather_list = [torch.zeros_like(samples) for _ in range(sp_size)] if local_rank == 0 else None
    #         torch.distributed.gather(samples, dst=src, gather_list=gather_list, group=mpu.get_sequence_parallel_group())
    #         if local_rank == 0:
    #             samples = torch.concat(gather_list, dim=chunk_dim)

    #     return samples

    @torch.no_grad()
    def log_conditionings(self, batch: Dict, n: int) -> Dict:
        """
        Defines heuristics to log different conditionings.
        These can be lists of strings (text-to-image), tensors, ints, ...
        """
        image_h, image_w = batch[self.input_key].shape[3:]
        log = dict()

        for embedder in self.conditioner.embedders:
            if (
                    (self.log_keys is None) or (embedder.input_key in self.log_keys)
            ) and not self.no_cond_log:
                x = batch[embedder.input_key][:n]
                if isinstance(x, torch.Tensor):
                    if x.dim() == 1:
                        # class-conditional, convert integer to string
                        x = [str(x[i].item()) for i in range(x.shape[0])]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 4)
                    elif x.dim() == 2:
                        # size and crop cond and the like
                        x = [
                            "x".join([str(xx) for xx in x[i].tolist()])
                            for i in range(x.shape[0])
                        ]
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                elif isinstance(x, (List, ListConfig)):
                    if isinstance(x[0], str):
                        # strings
                        xc = log_txt_as_img((image_h, image_w), x, size=image_h // 20)
                    else:
                        raise NotImplementedError()
                else:
                    raise NotImplementedError()
                log[embedder.input_key] = xc
        return log

    @torch.no_grad()
    def log_video(
            self,
            batch: Dict,
            N: int = 8,
            sample: bool = True,
            ucg_keys: List[str] = None,
            only_log_video_latents = False,
            **kwargs,
    ) -> Dict:
        raise NotImplementedError("log_video 需要重写")
        return None
    

    @classmethod
    def from_pretrained_base(cls, args=None, *, prefix='', build_only=False, overwrite_args={}, **kwargs):
        '''Load a pretrained checkpoint of the current model.
            Args:
                name: The identifier of the pretrained model.
                args: NameSpace. will add the loaded args into it. None will create a new model-only one with defaults.
                path: the parent folder of existing `name` model. Default: SAT_HOME.
                url: the url of the model. Default: SAT_URL.
                prefix: the prefix of the checkpoint. Default: ''.
            Returns:
                model: the loaded model.
                args: the loaded args.
        '''

        # create a new args if not provided
        if args is None:
            args = cls.get_args()
        args = overwrite_args_by_dict(args, overwrite_args=overwrite_args)
        model = get_model(args, cls, **kwargs)
        if not build_only:
            load_checkpoint(model, args, prefix=prefix)
        return model, deepcopy(args)
    
    @classmethod
    def from_pretrained(cls, args=None, *, prefix='', build_only=False, use_node_group=True, overwrite_args={}, **kwargs):
        if build_only or 'model_parallel_size' not in overwrite_args:
            return cls.from_pretrained_base(args=args, prefix=prefix, build_only=build_only, overwrite_args=overwrite_args, **kwargs)
        else:
            new_model_parallel_size = overwrite_args['model_parallel_size']
            if new_model_parallel_size != 1 or new_model_parallel_size == 1 and args.model_parallel_size == 1:
                model, model_args = cls.from_pretrained_base(args=args, prefix=prefix, build_only=True, overwrite_args=overwrite_args, **kwargs)
                local_rank = get_node_rank() if use_node_group else get_model_parallel_rank()
                world_size = torch.distributed.get_world_size()
                assert world_size % new_model_parallel_size == 0, "world size should be a multiplier of new model_parallel_size."
                destroy_model_parallel()
                initialize_model_parallel(1)
                if local_rank == 0:
                    args.use_gpu_initialization = False
                    args.device = 'cpu'
                    overwrite_args.pop('model_parallel_size')
                    model_full, args_ = cls.from_pretrained_base(args=args, prefix=prefix, build_only=False, overwrite_args=overwrite_args, **kwargs)
                    if args_.model_parallel_size != 1:
                        raise Exception("We do not support overwriting model_parallel_size when original model_parallel_size != 1. Try merging the model using `from_pretrained(xxx,overwrite_args={'model_parallel_size':1})` first if you still want to change model_parallel_size!")
                if hasattr(args, 'mode') and args.mode == 'inference': # For multi-node inference, we should prevent rank 0 eagerly printing some info.
                    torch.distributed.barrier()
                destroy_model_parallel()
                initialize_model_parallel(new_model_parallel_size)
                if local_rank == 0:
                    mp_split_model_rank0(model, model_full, use_node_group=use_node_group)
                    del model_full
                else:
                    mp_split_model_receive(model, use_node_group=use_node_group)
                reset_random_seed(6)
            else:
                overwrite_args.pop('model_parallel_size')
                model, model_args = cls.from_pretrained_base(args=args, prefix=prefix, build_only=False, overwrite_args=overwrite_args, **kwargs)
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
                assert world_size == model_args.model_parallel_size, "world size should be equal to model_parallel_size."
                destroy_model_parallel()
                initialize_model_parallel(1)
                if rank == 0:
                    args.use_gpu_initialization = False
                    args.device = 'cpu'
                    overwrite_args['model_parallel_size'] = 1
                    overwrite_args['model_config'] = args.model_config
                    overwrite_args['model_config']['network_config']['params']['transformer_args']['model_parallel_size'] = 1
                    model_full, args_ = cls.from_pretrained_base(args=args, prefix=prefix, build_only=True, overwrite_args=overwrite_args, **kwargs)
                torch.distributed.barrier()
                destroy_model_parallel()
                initialize_model_parallel(model_args.model_parallel_size)
                if rank == 0:
                    mp_merge_model_rank0(model, model_full)
                    model, model_args = model_full, args_
                else:
                    mp_merge_model_send(model)
                    model_args.model_parallel_size = 1
                destroy_model_parallel()
                initialize_model_parallel(1)
            return model, model_args
