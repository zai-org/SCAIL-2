from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import ListConfig
import math

from ...util import append_dims, instantiate_from_config
from ...modules.autoencoding.lpips.loss.lpips import LPIPS
#import rearrange
from einops import rearrange
import random
from sat import mpu

def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b

def time_shift(mu, t, mode=None):
    if mode == 'meta':
        return 1 / (1 + math.exp(mu)/t - math.exp(mu))
    elif mode == 'normal':
        return math.exp(mu) / (math.exp(mu) + 1 / t - 1)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def block_noise(ref_x, randn_like=torch.randn_like, block_size=1, device=None):
    """
    build block noise
    """
    g_noise = randn_like(ref_x)
    if block_size == 1:
        return g_noise

    blk_noise = torch.zeros_like(ref_x, device=device)
    for px in range(block_size):
        for py in range(block_size):
            blk_noise += torch.roll(g_noise, shifts=(px, py), dims=(-2, -1))

    blk_noise = blk_noise / block_size  # to maintain the same std on each pixel

    return blk_noise

def time_block_noise(ref_x, rankn_like=torch.randn_like, block_size=1, device=None):
    # ref_x [b,t,c,h,w]
    ref_x = ref_x.permute(0, 2, 3, 4, 1)
    g_noise = rankn_like(ref_x)
    blk_noise = torch.zeros_like(ref_x, device=device)
    for px in range(block_size):
        blk_noise += torch.roll(g_noise, shifts=px, dims=-1)
    blk_noise = blk_noise / math.sqrt(block_size)
    blk_noise = blk_noise.permute(0, 4, 1, 2, 3)
    return blk_noise


class StandardDiffusionLoss(nn.Module):
    def __init__(
        self,
        sigma_sampler_config,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.sigma_sampler = instantiate_from_config(sigma_sampler_config)

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        sigmas = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = noise + append_dims(
                torch.randn(input.shape[0]).to(input.device), input.ndim
            ) * self.offset_noise_level
            noise = noise.to(input.dtype)
        noised_input = input.float() + noise * append_dims(sigmas, input.ndim)
        model_output = denoiser(
            network, noised_input, sigmas, cond, **additional_model_inputs
        )
        w = append_dims(denoiser.w(sigmas), input.ndim)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


class VideoDiffusionLoss(StandardDiffusionLoss):
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, fixed_frames=0, **kwargs):
        self.fixed_frames = fixed_frames
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        if self.fixed_frames > 0:
            pre_input, input = input.split([self.fixed_frames, input.shape[1] - self.fixed_frames], dim=1)

            if "image" in batch.keys():
                # print("image in batch")
                fixed_input = batch["image"]
            else:
                fixed_input = pre_input

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        idx = idx.to(input.device)

        noise = torch.randn_like(input)

        #broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

        # torch.set_printoptions(threshold=10, edgeitems=2)
        # print("rank:", torch.distributed.get_rank(), "idx:", idx, "noise:", noise)
        # print("rank:", torch.distributed.get_rank(), "input", input)
        additional_model_inputs['idx'] = idx

        if self.offset_noise_level > 0.0:
            noise = noise + append_dims(
                torch.randn(input.shape[0]).to(input.device), input.ndim
            ) * self.offset_noise_level
        if self.block_scale is not None:
            noise = self.get_blk_noise(noise)

        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims((1-alphas_cumprod_sqrt**2)**0.5, input.ndim)
        if self.fixed_frames > 0:
            noised_input = torch.cat([fixed_input, noised_input], dim=1)

        if "concat_images" in batch.keys():
            additional_model_inputs["concat_images"] = batch["concat_images"]

        model_output = denoiser(
            network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs
        )
        if self.fixed_frames > 0:
            if "image" in batch.keys():
                input = torch.cat([pre_input, input], dim=1)
            else:
                model_output = model_output[:, self.fixed_frames:]
        w = append_dims(1/(1-alphas_cumprod_sqrt**2), input.ndim) # v-pred
        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)
        return self.get_loss(model_output, input, w)

    def get_blk_noise(self, noise):
        t = noise.shape[1]
        blk_noise = time_block_noise(noise, block_size=min(self.block_size, t), device=noise.device)
        noise = noise * (1 - self.block_scale ** 2) ** 0.5 + self.block_scale * blk_noise
        return noise

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


class Image2VideoDiffusionLoss(StandardDiffusionLoss):
    def __init__(self, block_scale=None, block_size=None, min_snr_value=None, **kwargs):
        self.block_scale = block_scale
        self.block_size = block_size
        self.min_snr_value = min_snr_value
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        idx = idx.to(input.device)

        noise = torch.randn_like(input)

        #broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

        # torch.set_printoptions(threshold=10, edgeitems=2)
        # print("rank:", torch.distributed.get_rank(), "idx:", idx, "noise:", noise)
        # print("rank:", torch.distributed.get_rank(), "input", input)
        additional_model_inputs['idx'] = idx

        if self.offset_noise_level > 0.0:
            noise = noise + append_dims(
                torch.randn(input.shape[0]).to(input.device), input.ndim
            ) * self.offset_noise_level
        if self.block_scale is not None:
            noise = self.get_blk_noise(noise)

        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims((1-alphas_cumprod_sqrt**2)**0.5, input.ndim)

        image = torch.concat([batch["image"], torch.zeros_like(noised_input[:, :, 1:])], dim=2)
        noised_input = torch.concat([noised_input, image], dim=2)
        model_output = denoiser(
            network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs
        )

        w = append_dims(1/(1-alphas_cumprod_sqrt**2), input.ndim) # v-pred
        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)
        return self.get_loss(model_output, input, w)

    def get_blk_noise(self, noise):
        t = noise.shape[1]
        blk_noise = time_block_noise(noise, block_size=min(self.block_size, t), device=noise.device)
        noise = noise * (1 - self.block_scale ** 2) ** 0.5 + self.block_scale * blk_noise
        return noise

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss

def get_3d_position_ids(frame_len, h, w):
    i = torch.arange(frame_len).view(frame_len, 1, 1).expand(frame_len, h, w)
    j = torch.arange(h).view(1, h, 1).expand(frame_len, h, w)
    k = torch.arange(w).view(1, 1, w).expand(frame_len, h, w)
    position_ids = torch.stack([i, j, k], dim=-1)
    return position_ids


class VideoDiffusionLossPnP(StandardDiffusionLoss):
    def __init__(self, block_scale=None, block_size=None, **kwargs):
        self.block_scale = block_scale
        self.block_size = block_size
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch):
        # make pack n pack
        bz = len(batch["fps"])

        txt_list = []
        for i in range(bz):
            txt_list.extend(batch["txt"][i])
        batch["txt"] = txt_list

        cond = conditioner(batch)

        max_length = 5500
        txt_length = 225
        patch_size = 2

        # input [b,t,c,h,w]
        h = input.shape[-2] // patch_size
        w = input.shape[-1] // patch_size
        num_frame_token = h * w
        input = rearrange(input, 'b t c (h p) (w q) -> b (t h w) (p q c)', p=2, q=2).squeeze(0) #[t * num_frame_tokens, hidden_size]
        # new input
        tokens = []
        txt_masks = []
        image_masks = []
        position_ids = []
        attention_masks = []
        frame_pos = 0
        for i in range(bz):
            txt_mask = torch.zeros([max_length], dtype=torch.bool, device=input.device)
            image_mask = torch.zeros([max_length], dtype=torch.bool, device=input.device)
            token = torch.zeros([max_length, input.shape[-1]], device=input.device)
            position_id = torch.ones([max_length, 3], dtype=torch.long, device=input.device) * -1
            attention_mask = torch.zeros([max_length, max_length], dtype=torch.bool, device=input.device)
            frame_lens = batch["frame_lens"][i]
            drop_ratios = batch["drop_ratios"][i]
            token_pos = 0
            for j, frame_len in enumerate(frame_lens):
                txt_mask[token_pos: token_pos+txt_length] = 1
                # token[token_pos: token_pos+txt_length] = txt_embedding[txt_pos]
                # txt_pos += 1
                token_pos += txt_length
                assert frame_len % 4 == 1, str(frame_len)
                frame_len = int((frame_len + 3)/4)
                tmp_position_ids = get_3d_position_ids(frame_len, h, w)

                if drop_ratios[j] > 0:
                    drop_num = int(frame_len * num_frame_token * drop_ratios[j])
                    sample_index = random.sample(list(range(frame_len * num_frame_token)), drop_num)
                    mask = torch.ones(frame_len * num_frame_token).bool()
                    mask[sample_index] = 0
                    tmp_position_ids = tmp_position_ids[mask]
                    num_sample_token = frame_len * num_frame_token - drop_num
                    insert_input = input[frame_pos * num_frame_token:(frame_pos+frame_len)*num_frame_token, :][mask]
                else:
                    num_sample_token = frame_len * num_frame_token
                    insert_input = input[frame_pos * num_frame_token:(frame_pos+frame_len)*num_frame_token, :]
                attention_mask[token_pos-txt_length: token_pos + num_sample_token, token_pos-txt_length: token_pos + num_sample_token] = 1
                image_mask[token_pos: token_pos + num_sample_token] = 1
                position_id[token_pos: token_pos + num_sample_token] = tmp_position_ids
                token[token_pos: token_pos + num_sample_token] = insert_input
                token_pos += num_sample_token
                frame_pos += frame_len
            tokens.append(token)
            txt_masks.append(txt_mask)
            image_masks.append(image_mask)
            position_ids.append(position_id)
            attention_masks.append(attention_mask)
        tokens = torch.stack(tokens)
        txt_masks = torch.stack(txt_masks)
        image_masks = torch.stack(image_masks)
        position_ids = torch.stack(position_ids)
        attention_masks = torch.stack(attention_masks).unsqueeze(1)
        input = tokens
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }
        additional_model_inputs["txt_mask"] = txt_masks
        additional_model_inputs["image_mask"] = image_masks
        additional_model_inputs["rope_position_ids"] = position_ids
        additional_model_inputs["attention_mask"] = attention_masks

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[0], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        idx = idx.to(input.device)
        noise = torch.randn_like(input)
        if self.offset_noise_level > 0.0:
            noise = noise + append_dims(
                torch.randn(input.shape[0]).to(input.device), input.ndim
            ) * self.offset_noise_level
        if self.block_scale is not None:
            noise = self.get_blk_noise(noise)


        #broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())

        additional_model_inputs['idx'] = idx

        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims((1-alphas_cumprod_sqrt**2)**0.5, input.ndim) * image_masks.float().unsqueeze(-1)
        model_output = denoiser(network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs)
        w = append_dims(1/(1-alphas_cumprod_sqrt**2), input.ndim) # v-pred
        return self.get_loss(model_output, input, w, image_masks.float().unsqueeze(-1))

    def get_blk_noise(self, noise):
        t = noise.shape[1]
        blk_noise = time_block_noise(noise, block_size=min(self.block_size, t), device=noise.device)
        noise = noise * (1 - self.block_scale ** 2) ** 0.5 + self.block_scale * blk_noise
        return noise

    def get_loss(self, model_output, target, w, loss_mask=None):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2 * loss_mask).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()) * loss_mask.reshape(target.shape[0], -1), 1
            )


class PDDiffusionLoss(nn.Module):
    def __init__(
        self,
        type="l2",
        offset_noise_level=0.0,
        batch2model_keys: Optional[Union[str, List[str], ListConfig]] = None,
        discretization_config=None,
        num_idx=None,
        add_dsm_loss=False
    ):
        super().__init__()

        assert type in ["l2", "l1", "lpips"]

        self.type = type
        self.offset_noise_level = offset_noise_level

        if type == "lpips":
            self.lpips = LPIPS().eval()

        if not batch2model_keys:
            batch2model_keys = []

        if isinstance(batch2model_keys, str):
            batch2model_keys = [batch2model_keys]

        self.batch2model_keys = set(batch2model_keys)
        self.num_idx = num_idx
        self.alpha_cumprod_sqrt_all, timesteps = instantiate_from_config(discretization_config)(
            num_idx, do_append_zero=False, flip=True, return_idx=True
        )

        self.alpha_cumprod_sqrt_all = torch.cat([self.alpha_cumprod_sqrt_all.new_ones([1]), self.alpha_cumprod_sqrt_all])
        self.timesteps = torch.cat([torch.tensor(list(timesteps)).new_zeros([1])-1, torch.tensor(list(timesteps))])

        self.add_dsm_loss = add_dsm_loss
        print(f'add dsm loss: {add_dsm_loss}')
    
    def __call__(self, network, denoiser, conditioner, input, batch, teacher_model, sampler):
        self.alpha_cumprod_sqrt_all = self.alpha_cumprod_sqrt_all.to(input.device)
        self.timesteps = self.timesteps.to(input.device)

        uncond = None
        # cond = conditioner(batch)
        cond, uncond = conditioner.get_unconditional_conditioning(batch, force_uc_zero_embeddings=['txt'])
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        scale = (1.5 + torch.rand((input.shape[0],)) * 7.5).to(input.device)
        scale_emb = guidance_scale_embedding(scale, 512).to(input.dtype)

        rand = torch.randint(1, self.num_idx//2+1, (input.shape[0],))*2
        alphas_cumprod_sqrt = self.alpha_cumprod_sqrt_all[rand].to(input.device)
        alphas_cumprod_sqrt_next = self.alpha_cumprod_sqrt_all[rand-1].to(input.device)
        alphas_cumprod_sqrt_next_next = self.alpha_cumprod_sqrt_all[rand-2].to(input.device)
        noise = torch.randn_like(input)

        rand = rand.to(input.device)
        if mpu.get_model_parallel_world_size() > 1:
            #broadcast noise
            mp_size = mpu.get_model_parallel_world_size()
            global_rank = torch.distributed.get_rank() // mp_size
            src = global_rank * mp_size
            torch.distributed.broadcast(rand, src=src, group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(scale, src=src, group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(scale_emb, src=src, group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(alphas_cumprod_sqrt_next, src=src, group=mpu.get_model_parallel_group())
            torch.distributed.broadcast(alphas_cumprod_sqrt_next_next, src=src, group=mpu.get_model_parallel_group())

        additional_model_inputs['idx'] = self.timesteps[rand]
        additional_model_inputs['scale_emb'] = scale_emb

        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims((1-alphas_cumprod_sqrt**2)**0.5, input.ndim)
        model_output = denoiser(
            network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs
        )
        
        with torch.no_grad():
            sample_denoiser = lambda input, sigma, c, **additional_model_inputs: denoiser(teacher_model, input, sigma, c, **additional_model_inputs)

            from ...modules.diffusionmodules.sampling import VideoDDIMSampler, VPSDEDPMPP2MSampler
            if isinstance(sampler, VideoDDIMSampler):
                x_next = sampler.sampler_step(alphas_cumprod_sqrt, alphas_cumprod_sqrt_next, sample_denoiser, noised_input, cond, uncond, idx=rand, timestep=self.timesteps[rand], scale=1, scale_emb=scale_emb)
                x_next_next = sampler.sampler_step(alphas_cumprod_sqrt_next, alphas_cumprod_sqrt_next_next, sample_denoiser, x_next, cond, uncond, idx=rand-1, timestep=self.timesteps[rand-1], scale=1, scale_emb=scale_emb)
                a_t = (1-alphas_cumprod_sqrt_next_next**2)**0.5 / (1-alphas_cumprod_sqrt**2)**0.5
                target = (x_next_next - append_dims(a_t, input.ndim) * noised_input) / append_dims((alphas_cumprod_sqrt_next_next - a_t*alphas_cumprod_sqrt), input.ndim)

        w = append_dims(1/(1-alphas_cumprod_sqrt**2), input.ndim) # v-pred
        pd_loss = self.get_loss(model_output, target, w).mean()
        loss = pd_loss
        if self.add_dsm_loss:
            dsm_loss = self.get_loss(model_output, input, w).mean()
            loss += 0.001 * dsm_loss

        return loss

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss

def guidance_scale_embedding(w, embedding_dim=512, dtype=torch.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`torch.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb).to(w.device).to(w.dtype)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb

class RFLossAmp(StandardDiffusionLoss):
    def __init__(self, schedule_shift=False, **kwargs):
        super().__init__(**kwargs)
        self.schedule_shift = schedule_shift

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        sigma = self.sigma_sampler(input.shape[0]).to(input.device)
        noise = torch.randn_like(input)
        # input [b,t,c,h,w]
        if self.schedule_shift:
            for index in range(sigma.shape[0]):
                image_seq_len = input.shape[-1] * input.shape[-2] // network.diffusion_model.patch_size[-1] // network.diffusion_model.patch_size[-2]
                mu = get_lin_function(y1=0.5, y2=1.15)(image_seq_len)
                sigma[index] = time_shift(mu, sigma[index], mode='normal')

        #broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        sp_size = mpu.get_sequence_parallel_world_size()
        if mp_size > 1 or sp_size > 1:
            src = mpu.get_data_broadcast_src_rank()
            torch.distributed.broadcast(noise, src=src, group=mpu.get_data_broadcast_group())
            torch.distributed.broadcast(sigma, src=src, group=mpu.get_data_broadcast_group())
        
        chunk_dim = None
        if sp_size > 1:
            sp_rank = mpu.get_sequence_parallel_rank()
            h, w = noise.shape[-2:]
            if h < w:
                chunk_dim = 3
            else:
                chunk_dim = 4
            noise = torch.chunk(noise, sp_size, dim=chunk_dim)[sp_rank] # TODO: 打一下noise的shape
            input = torch.chunk(input, sp_size, dim=chunk_dim)[sp_rank]
            additional_model_inputs['chunk_dim'] = chunk_dim

            if "concat_images" in batch.keys():
                batch["concat_images"] = torch.chunk(batch["concat_images"], sp_size, dim=chunk_dim)[sp_rank]
            if 'ref_concat' in batch.keys():
                batch['ref_concat'] = torch.chunk(batch['ref_concat'], sp_size, dim=chunk_dim)[sp_rank]
            if "concat_smpl_render" in batch.keys():
                batch['concat_smpl_render'] = torch.chunk(batch['concat_smpl_render'], sp_size, dim=chunk_dim)[sp_rank]
            if "concat_cheek_hands" in batch.keys():
                batch['concat_cheek_hands'] = torch.chunk(batch['concat_cheek_hands'], sp_size, dim=chunk_dim)[sp_rank]
        
        if "concat_images" in batch.keys():
            additional_model_inputs["concat_images"] = batch["concat_images"]
        if "image_clip_features" in batch.keys():
            additional_model_inputs["image_clip_features"] = batch["image_clip_features"]
        if 'ref_concat' in batch.keys():
            additional_model_inputs['ref_concat'] = batch['ref_concat']
        if 'concat_smpl_render' in batch.keys():
            additional_model_inputs['concat_smpl_render'] = batch['concat_smpl_render']
        if 'concat_cheek_hands' in batch.keys():
            additional_model_inputs['concat_cheek_hands'] = batch['concat_cheek_hands']

        noised_input = input.float() * append_dims(1 - sigma, input.ndim) + noise * append_dims(sigma, input.ndim)
        model_output = denoiser(
            network, noised_input, sigma, cond, **additional_model_inputs
        )
    
        latent_hands_mask = batch["latent_hands_mask"]
        latent_faces_mask = batch["latent_faces_mask"]   # 脸部为1，其它区域为0
        if sp_size > 1:
            latent_hands_mask = torch.chunk(latent_hands_mask, sp_size, dim=chunk_dim)[sp_rank]
            latent_faces_mask = torch.chunk(latent_faces_mask, sp_size, dim=chunk_dim)[sp_rank]

        # 计算区域损失：使用weight_mask
        weight_mask = torch.ones_like(model_output) + 0.5 * latent_hands_mask + 0.5 * latent_faces_mask
        amp_loss = self.get_loss(model_output, noise - input, weight_mask)

        return amp_loss


    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss
        

class RFLoss(StandardDiffusionLoss):
    def __init__(self, schedule_shift=False, **kwargs):
        super().__init__(**kwargs)
        self.schedule_shift = schedule_shift

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        sigma = self.sigma_sampler(input.shape[0]).to(input.device)

        noise = torch.randn_like(input)
        # input [b,t,c,h,w]
        if self.schedule_shift:
            for index in range(sigma.shape[0]):
                image_seq_len = input.shape[-1] * input.shape[-2] // network.diffusion_model.patch_size[-1] // network.diffusion_model.patch_size[-2]
                mu = get_lin_function(y1=0.5, y2=1.15)(image_seq_len)
                sigma[index] = time_shift(mu, sigma[index], mode='normal')

        #broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        sp_size = mpu.get_sequence_parallel_world_size()
        if mp_size > 1 or sp_size > 1:
            src = mpu.get_data_broadcast_src_rank()
            torch.distributed.broadcast(noise, src=src, group=mpu.get_data_broadcast_group())
            torch.distributed.broadcast(sigma, src=src, group=mpu.get_data_broadcast_group())
        
        chunk_dim = None
        if sp_size > 1:
            sp_rank = mpu.get_sequence_parallel_rank()
            h, w = noise.shape[-2:]
            if h < w:
                chunk_dim = 3
            else:
                chunk_dim = 4
            noise = torch.chunk(noise, sp_size, dim=chunk_dim)[sp_rank] # TODO: 打一下noise的shape
            # 给mask区域的noise置0即可
            input = torch.chunk(input, sp_size, dim=chunk_dim)[sp_rank]
            additional_model_inputs['chunk_dim'] = chunk_dim

            if "concat_images" in batch.keys():
                batch["concat_images"] = torch.chunk(batch["concat_images"], sp_size, dim=chunk_dim)[sp_rank]
            if 'ref_concat' in batch.keys():
                batch['ref_concat'] = torch.chunk(batch['ref_concat'], sp_size, dim=chunk_dim)[sp_rank]
            if "concat_smpl_render" in batch.keys():
                batch['concat_smpl_render'] = torch.chunk(batch['concat_smpl_render'], sp_size, dim=chunk_dim)[sp_rank]
            if "concat_latent_sam" in batch.keys():
                batch['concat_latent_sam'] = torch.chunk(batch['concat_latent_sam'], sp_size, dim=chunk_dim)[sp_rank]
            if "concat_latent_ref_mask" in batch.keys():
                batch['concat_latent_ref_mask'] = torch.chunk(batch['concat_latent_ref_mask'], sp_size, dim=chunk_dim)[sp_rank]
            if "concat_cheek_hands" in batch.keys():
                batch['concat_cheek_hands'] = torch.chunk(batch['concat_cheek_hands'], sp_size, dim=chunk_dim)[sp_rank]
            if "history_mask" in batch.keys():
                batch['history_mask'] = torch.chunk(batch['history_mask'], sp_size, dim=chunk_dim)[sp_rank]
            if "latent_ocr_mask" in batch.keys():
                batch['latent_ocr_mask'] = torch.chunk(batch['latent_ocr_mask'], sp_size, dim=chunk_dim)[sp_rank]

        if "concat_images" in batch.keys():
            additional_model_inputs["concat_images"] = batch["concat_images"]
        if "image_clip_features" in batch.keys():
            additional_model_inputs["image_clip_features"] = batch["image_clip_features"]
        if 'ref_concat' in batch.keys():
            additional_model_inputs['ref_concat'] = batch['ref_concat']
        if 'concat_smpl_render' in batch.keys():
            additional_model_inputs['concat_smpl_render'] = batch['concat_smpl_render']
        if "concat_latent_sam" in batch.keys():
            additional_model_inputs['concat_latent_sam'] = batch['concat_latent_sam']
        if "concat_latent_ref_mask" in batch.keys():
            additional_model_inputs['concat_latent_ref_mask'] = batch['concat_latent_ref_mask']
        if 'concat_cheek_hands' in batch.keys():
            additional_model_inputs['concat_cheek_hands'] = batch['concat_cheek_hands']
        if 'history_mask' in batch.keys():
            additional_model_inputs['history_mask'] = batch['history_mask']
        if 'ref_mask_flag' in batch.keys():
            additional_model_inputs['ref_mask_flag'] = batch['ref_mask_flag']

        history_mask = batch['history_mask']  # b t 4 h w
        # 将history_mask从[b, t, 4, h, w]扩展到[b, t, c, h, w]以匹配noise和input的维度
        # history_mask中1表示历史帧区域，0表示生成帧区域
        c = input.shape[2]
        history_mask_expanded = history_mask[:, :, :1, :, :].expand(-1, -1, c, -1, -1)  # b t c h w
        
        # 计算noised_input
        # 对于生成帧（history_mask=0）: noised_input = (1-sigma)*input + sigma*noise
        # 对于历史帧（history_mask=1）: noised_input = input（保持清晰，不受sigma影响）
        noised_input_generated = input.float() * append_dims(1 - sigma, input.ndim) + noise * append_dims(sigma, input.ndim)
        noised_input = history_mask_expanded * input.float() + (1 - history_mask_expanded) * noised_input_generated

        model_output = denoiser(
            network, noised_input, sigma, cond, **additional_model_inputs
        )

        latent_ocr_mask = batch["latent_ocr_mask"]  # shape和latent_hands_mask/latent_faces_mask一样，是和latent对齐的，ocr部分为1
        latent_ocr_mask_expanded = latent_ocr_mask[:, :, :1, :, :].expand(-1, -1, c, -1, -1)  # b t c h w

        # 只对非历史帧、非ocr区域计算loss
        loss_mask = (1 - history_mask_expanded) * (1 - latent_ocr_mask_expanded)
        # loss_mask = 1 - history_mask_expanded
        return self.get_loss(model_output, noise - input, loss_mask)
    

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


class RFLoss_meta(StandardDiffusionLoss):
    def __init__(self, flow_sigma_min=None, schedule_shift=False, **kwargs):
        self.flow_sigma_min = flow_sigma_min
        self.schedule_shift = schedule_shift
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch):
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        t_indices = self.sigma_sampler(input.shape[0]).to(input.device)
        # input [b,t,c,h,w]
        if self.schedule_shift:
            for index in range(t_indices.shape[0]):
                image_seq_len = input.shape[-1] * input.shape[-2] // network.diffusion_model.patch_size[-1] // network.diffusion_model.patch_size[-2]
                mu = get_lin_function(y1=0.5, y2=1.15)(image_seq_len)
                t_indices[index] = time_shift(mu, t_indices[index], mode='meta')

        noise = torch.randn_like(input)

        # broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        sp_size = mpu.get_sequence_parallel_world_size()
        if mp_size > 1 or sp_size > 1:
            src = mpu.get_data_broadcast_src_rank()
            torch.distributed.broadcast(noise, src=src, group=mpu.get_data_broadcast_group())
            torch.distributed.broadcast(t_indices, src=src, group=mpu.get_data_broadcast_group())

        if sp_size > 1:
            sp_rank = mpu.get_sequence_parallel_rank()
            h, w = input.shape[-2:]
            if h < w:
                chunk_dim = 3
            else:
                chunk_dim = 4
            noise = torch.chunk(noise, sp_size, dim=chunk_dim)[sp_rank]
            input = torch.chunk(input, sp_size, dim=chunk_dim)[sp_rank]
            additional_model_inputs['chunk_dim'] = chunk_dim
            if "concat_images" in batch.keys():
                batch["concat_images"] = torch.chunk(batch["concat_images"], sp_size, dim=chunk_dim)[sp_rank]

        if "concat_images" in batch.keys():
            additional_model_inputs["concat_images"] = batch["concat_images"]
        noised_input = input.float() * append_dims(t_indices, input.ndim) + noise * append_dims(1 - (1 - self.flow_sigma_min) * t_indices, input.ndim)
        model_output = denoiser(
            network, noised_input, t_indices, cond, **additional_model_inputs
        )
        return self.get_loss(model_output, input - (1 - self.flow_sigma_min) * noise, 1)

    def get_loss(self, model_output, target, w):
        if self.type == "l2":
            return torch.mean(
                (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
            )
        elif self.type == "l1":
            return torch.mean(
                (w * (model_output - target).abs()).reshape(target.shape[0], -1), 1
            )
        elif self.type == "lpips":
            loss = self.lpips(model_output, target).reshape(-1)
            return loss


class TASDLoss(StandardDiffusionLoss):
    def __init__(self, min_snr_value=None, **kwargs):
        self.min_snr_value = min_snr_value
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch):
        # input b, t, d, h, w
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        alphas_cumprod_sqrt, idx = self.sigma_sampler(input.shape[:2], return_idx=True)
        alphas_cumprod_sqrt = alphas_cumprod_sqrt.to(input.device)
        idx = idx.to(input.device)

        noise = torch.randn_like(input)

        #broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(idx, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(alphas_cumprod_sqrt, src=src, group=mpu.get_model_parallel_group())


        noised_input = input.float() * append_dims(alphas_cumprod_sqrt, input.ndim) + noise * append_dims((1-alphas_cumprod_sqrt**2)**0.5, input.ndim)

        noised_input = torch.concat([input, noised_input], dim=1)
        alphas_cumprod_sqrt = torch.cat([torch.ones_like(alphas_cumprod_sqrt), alphas_cumprod_sqrt], dim=1)

        idx = torch.cat([torch.zeros_like(idx), idx], dim=1)
        additional_model_inputs['idx'] = idx
        # get position ids
        patch_size = network.diffusion_model.patch_size
        position_ids = get_3d_position_ids(input.shape[1]//patch_size[0], input.shape[3]//patch_size[1], input.shape[4]//patch_size[2]).reshape(-1, 3)
        position_ids = position_ids.repeat([2, 1])
        position_ids = position_ids.unsqueeze(0).expand(input.shape[0], -1, -1)
        additional_model_inputs['rope_position_ids'] = position_ids
        model_output = denoiser(
            network, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs
        )
        model_output = model_output[:, input.shape[1]:]
        alphas_cumprod_sqrt = alphas_cumprod_sqrt[:, input.shape[1]:]
        w = append_dims(1/(1-alphas_cumprod_sqrt**2), input.ndim) # v-pred

        if self.min_snr_value is not None:
            w = min(w, self.min_snr_value)
        return self.get_loss(model_output, input, w)

    def get_loss(self, model_output, target, w):
        return torch.mean(
            (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
        )


class TASDLoss_RF(StandardDiffusionLoss):
    def __init__(self, schedule_shift=False, noise_augmentation=False, aug=False, aug_max=None, remove_first=True, **kwargs):
        self.schedule_shift = schedule_shift
        self.noise_augmentation = noise_augmentation
        self.aug = aug
        self.aug_max = aug_max
        self.remove_first = remove_first
        super().__init__(**kwargs)

    def __call__(self, network, denoiser, conditioner, input, batch):
        # input b, t, d, h, w
        cond = conditioner(batch)
        additional_model_inputs = {
            key: batch[key] for key in self.batch2model_keys.intersection(batch)
        }

        t_indices = self.sigma_sampler(input.shape[:2])
        t_indices = t_indices.to(input.device)

        noise = torch.randn_like(input)

        #broadcast noise
        mp_size = mpu.get_model_parallel_world_size()
        global_rank = torch.distributed.get_rank() // mp_size
        src = global_rank * mp_size
        torch.distributed.broadcast(noise, src=src, group=mpu.get_model_parallel_group())
        torch.distributed.broadcast(t_indices, src=src, group=mpu.get_model_parallel_group())

        def scale_shift(indices):
            for index in range(indices.shape[0]):
                image_seq_len = input.shape[-1] * input.shape[-2] // network.diffusion_model.patch_size[-1] // network.diffusion_model.patch_size[-2]
                mu = get_lin_function(y1=0.5, y2=1.15)(image_seq_len)
                indices[index] = time_shift(mu, indices[index], mode='normal')
            return indices

        if self.schedule_shift:
            t_indices = scale_shift(t_indices)

        scaled_input = input.float() * append_dims(1 - t_indices, input.ndim)
        scaled_noise = noise * append_dims(t_indices, input.ndim)

        noised_input = scaled_input + scaled_noise
        if self.noise_augmentation:
            input = input + torch.exp(torch.normal(mean=-3.0, std=0.5, size=input.shape).to(input.device)).to(input.dtype) * torch.randn_like(input)

        if not self.aug:
            noised_input = torch.concat([input, noised_input], dim=1)
            t_indices_input = torch.cat([torch.zeros_like(t_indices), t_indices], dim=1)
        else:
            aug_noise = torch.randn_like(input)
            aug_indices = torch.rand(input.shape[:2]) * self.aug_max
            aug_indices = aug_indices.to(input.device)
            aug_input = input.float() * append_dims(1 - aug_indices, input.ndim) + aug_noise * append_dims(aug_indices, input.ndim)
            noised_input = torch.concat([aug_input, noised_input], dim=1)
            t_indices_input = torch.cat([aug_indices, t_indices], dim=1)

        # get position ids
        patch_size = network.diffusion_model.patch_size
        position_ids = get_3d_position_ids(input.shape[1]//patch_size[0], input.shape[3]//patch_size[1], input.shape[4]//patch_size[2]).reshape(-1, 3)
        position_ids = position_ids.repeat([2, 1])
        position_ids = position_ids.unsqueeze(0).expand(input.shape[0], -1, -1)
        additional_model_inputs['rope_position_ids'] = position_ids
        model_output = denoiser(
            network, noised_input, t_indices_input, cond, **additional_model_inputs
        )
        model_output = model_output[:, input.shape[1]:]

        model_label = noise - input
        if self.remove_first:
            model_output = model_output[:, 1:]
            model_label = model_label[:, 1:]

        # loss = (model_output - model_label) ** 2
        # loss = torch.mean(loss, dim=[0, 2, 3, 4])
        # print(loss[0], torch.sum(loss[1:]), loss.shape)
        # breakpoint()
        return self.get_loss(model_output, model_label, 1)

    def get_loss(self, model_output, target, w):
        return torch.mean(
            (w * (model_output - target) ** 2).reshape(target.shape[0], -1), 1
        )
