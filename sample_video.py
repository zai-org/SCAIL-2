import os
import sys
import math
import argparse
import json
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
from PIL import Image
import imageio
import time
import gc
import copy
import torch
import numpy as np
from einops import rearrange, repeat
from torchvision.utils import make_grid
import torchvision.transforms as TT

from sgm.util import get_obj_from_str, isheatmap, exists

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat import mpu

import diffusion_video
from arguments import get_args, process_config_to_args
import decord
from decord import VideoReader
from torchvision import transforms
import shutil
import torch.nn.functional as F
from data_video import pad_last_frame, resize_for_rectangle_crop

def load_image_to_tensor_chw_normalized(image_data):
    # Open image using PIL
    image = Image.open(image_data).convert('RGB')  # Convert to RGB in case it's a grayscale image or has an alpha channel
    # Define a transform to convert image to tensor
    transform = TT.Compose([TT.ToTensor()])
    # Apply the transform
    image_tensor = transform(image)
    # Scale the tensor back to [0, 255] and convert to uint8 (decord does this too)
    image_tensor = (image_tensor * 2 - 1).unsqueeze(0)  # 1 C H W, -1-1
    # C H W
    return image_tensor


def downsample_and_compress_sam_to_latent_tchw(smpl_sam, temporal_compression_stride=4, use_downsample_before_compress=True):
    """将 binary sam mask (T, C, H, W) 先 0.5x 下采样再压缩到 VAE latent shape。
    空间用 nearest（与 wan-animate 一致）；时间用 reshape：首帧 repeat stride 次后 cat 剩余帧，view 成 C*stride 通道。
    输出形状 (C*stride, T_latent, H_latent, W_latent)，与 wan-animate get_i2v_mask 对齐，C=3 → 12通道。
    """
    T, C, H, W = smpl_sam.shape
    T_latent = (T - 1) // temporal_compression_stride + 1
    if use_downsample_before_compress:
        H_latent, W_latent = H // 2, W // 2
    else:
        H_latent, W_latent = H, W
    for _ in range(3):                   # VAE 3x 空间压缩
        H_latent = (H_latent + 1) // 2
        W_latent = (W_latent + 1) // 2
    out = F.interpolate(smpl_sam.float(), size=(H_latent, W_latent), mode='area')                                 # (T, C, H_latent, W_latent); area=avg-pool，比 nearest 更适合大倍率下采样，保留 0~1 覆盖比例
    out = (out * 4).round() / 4                                                                                    # quantize to {0, 0.25, 0.5, 0.75, 1}
    out = torch.cat([out[:1].repeat(temporal_compression_stride, 1, 1, 1), out[1:]], dim=0)                       # (T_latent*stride, C, H_latent, W_latent)
    out = out.view(T_latent, temporal_compression_stride * C, H_latent, W_latent)
    return out


_MASK_COLORS = torch.tensor([
    [1, 1, 1],  # white
    [1, 0, 0],  # red
    [0, 1, 0],  # green
    [0, 0, 1],  # blue
    [1, 1, 0],  # yellow
    [1, 0, 1],  # magenta
    [0, 1, 1],  # cyan
], dtype=torch.float32)  # (7, 3)


def extract_and_compress_mask_to_latent(mask_cthw, additional_spatial_downsample=1, temporal_compression_stride=4):
    """将 3通道 RGB 分割mask 转换为 28通道二值 latent，不经过 VAE。
    输入: (3, T, H, W)，值域 [-1, 1]
    输出: (28, T_latent, H_latent, W_latent)，值域 {0, 1}
    """
    C, T, H, W = mask_cthw.shape
    _ON_THRESH = (225.0 - 127.5) / 127.5  # ≈ 0.765，原始像素值 ≥ 225 才算"亮"
    mask = mask_cthw.permute(1, 0, 2, 3).float()  # (T, 3, H, W)
    R = (mask[:, 0:1] > _ON_THRESH).float()
    G = (mask[:, 1:2] > _ON_THRESH).float()
    B = (mask[:, 2:3] > _ON_THRESH).float()
    nR, nG, nB = 1 - R, 1 - G, 1 - B
    binary_7ch = torch.cat([
        R * G * B, R * nG * nB, nR * G * nB, nR * nG * B,
        R * G * nB, R * nG * B, nR * G * B,
    ], dim=1)  # (T, 7, H, W)
    _color_names = ['white', 'red', 'green', 'blue', 'yellow', 'magenta', 'cyan']
    _total = H * W * T
    for _i, _name in enumerate(_color_names):
        _ratio = binary_7ch[:, _i].sum().item() / _total
        if _ratio > 0.001:
            print(f"  [mask debug] ch{_i} {_name}: {_ratio:.4f} ({_ratio*100:.2f}%)")
    H_lat, W_lat = H, W
    if additional_spatial_downsample > 1:
        H_lat = H_lat // additional_spatial_downsample
        W_lat = W_lat // additional_spatial_downsample
    for _ in range(3):
        H_lat = (H_lat + 1) // 2
        W_lat = (W_lat + 1) // 2
    binary_7ch = F.interpolate(binary_7ch, size=(H_lat, W_lat), mode='area')  # area=均值下采样，完整保留覆盖比例
    T_latent = (T - 1) // temporal_compression_stride + 1
    padded = torch.cat([binary_7ch[:1].repeat(temporal_compression_stride, 1, 1, 1), binary_7ch[1:]], dim=0)
    out = padded.view(T_latent, temporal_compression_stride * 7, H_lat, W_lat).permute(1, 0, 2, 3)
    return out  # (28, T_latent, H_lat, W_lat)


def load_video_for_pose_sample(video_data):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=-1, width=-1)
    indices = np.arange(0, len(vr))
    temp_frms = vr.get_batch(indices)
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    return tensor_frms


import random
import numpy as np
import torch
from decord import VideoReader
from PIL import Image
import cv2

def find_file_with_patterns(directory, patterns):
    """Find file matching any of the given patterns in the directory"""
    for pattern in patterns:
        file_path = os.path.join(directory, pattern)
        if os.path.exists(file_path):
            return file_path
    return None

def read_from_cli():
    cnt = 0
    try:
        while True:
            x = input('Please input in format like <prompt>@@<example_dir>, e.g. the girl is dancing@@examples/001 (Ctrl-D quit): ')
            yield x.strip(), cnt
            cnt += 1
    except EOFError as e:
        pass

def read_from_file(p, rank=0, world_size=1):
    with open(p, 'r') as fin:
        cnt = -1
        for l in fin:
            cnt += 1
            if cnt % world_size != rank:
                continue
            yield l.strip(), cnt

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def get_batch(keys, value_dict, N: Union[List, ListConfig], T=None, device="cuda"):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = (
                np.repeat([value_dict["prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
            batch_uc["txt"] = (
                np.repeat([value_dict["negative_prompt"]], repeats=math.prod(N))
                .reshape(N)
                .tolist()
            )
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = (
                torch.tensor([value_dict["orig_height"], value_dict["orig_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = (
                torch.tensor(
                    [value_dict["crop_coords_top"], value_dict["crop_coords_left"]]
                )
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = (
                torch.tensor([value_dict["aesthetic_score"]]).to(device).repeat(*N, 1)
            )
            batch_uc["aesthetic_score"] = (
                torch.tensor([value_dict["negative_aesthetic_score"]])
                .to(device)
                .repeat(*N, 1)
            )

        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = (
                torch.tensor([value_dict["target_height"], value_dict["target_width"]])
                .to(device)
                .repeat(*N, 1)
            )
        elif key == "fps":
            batch[key] = (
                torch.tensor([value_dict["fps"]]).to(device).repeat(math.prod(N))
            )
        elif key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]]).to(device).repeat(math.prod(N))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(math.prod(N))
            )
        elif key == "pool_image":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=math.prod(N)).to(
                device, dtype=torch.half
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to("cuda"),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames":
            batch[key] = repeat(value_dict["cond_frames"], "1 ... -> b ...", b=N[0])
        elif key == "cond_frames_without_noise":
            batch[key] = repeat(
                value_dict["cond_frames_without_noise"], "1 ... -> b ...", b=N[0]
            )
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

def save_multi_video_grid_and_mp4(
    video_batches: list, save_dir: str, fps: int = 5, args=None, key=None
):
    os.makedirs(save_dir, exist_ok=True)
    # base_count = len(glob(os.path.join(save_path, "*.mp4")))
    multi_video_batch = torch.stack(video_batches, dim=2)
    for i, multi_vid in enumerate(multi_video_batch):
        # save_image(vid, fp=os.path.join(save_path, f"{base_count:06d}.png"), nrow=4)
        # multi_vid: T, N, c, h, w 
        gif_frames = []
        for multi_frame in multi_vid:
            frame = rearrange(multi_frame, "n c h w -> h (n w) c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_dir, f"{key}_{i:06d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)


def save_video_as_grid_and_mp4(
    video_batch: torch.Tensor, save_path: str, fps: int = 5, args=None, key=None
):
    os.makedirs(save_path, exist_ok=True)
    # base_count = len(glob(os.path.join(save_path, "*.mp4")))

    for i, vid in enumerate(video_batch):
        # save_image(vid, fp=os.path.join(save_path, f"{base_count:06d}.png"), nrow=4)
        gif_frames = []
        for frame in vid:
            frame = rearrange(frame, "c h w -> h w c")
            frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
            gif_frames.append(frame)
        now_save_path = os.path.join(save_path, f"{i:06d}.mp4")
        with imageio.get_writer(now_save_path, fps=fps) as writer:
            for frame in gif_frames:
                writer.append_data(frame)

def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls
    if args.load is not None:
        load_checkpoint(model, args)
    model.eval()

    if args.input_type == 'cli':
        assert mpu.get_data_parallel_world_size() == 1, 'Only dp = 1 supported in cli mode.'
        data_iter = read_from_cli()
    elif args.input_type == 'txt':
        dp_rank, dp_world_size = mpu.get_data_parallel_rank(), mpu.get_data_parallel_world_size()
        data_iter = read_from_file(args.input_file, rank=dp_rank, world_size=dp_world_size)
    else:
        raise NotImplementedError
    sample_func = model.sample
    
    # if not args.multi_cond_cfg:
    #     sample_func = model.sample
    # else:
    #     sample_func = model.sample_with_pose_cond

    num_samples = [1]
    force_uc_zero_embeddings = []

    vae_compress_size = args.vae_compress_size
    print('VAE_compress_size:', vae_compress_size)
    # if args.image2video:
    #     zero_pad_dict = torch.load('zero_pad_dict.pt', map_location='cpu')

    with torch.no_grad():
        torch.distributed.barrier(group=mpu.get_data_broadcast_group())
        while True:
            stopped = False
            if mpu.get_data_broadcast_rank() == 0:
                try:
                    text, cnt = next(data_iter)
                except StopIteration:
                    text = ''
                    stopped = True

                # text = 'FPS-%d. ' % args.sampling_fps + text

            else:
                text = ''
                cnt = 0

            broadcast_list = [text, cnt, stopped]
            # broadcast
            mp_size = mpu.get_model_parallel_world_size()
            sp_size = mpu.get_sequence_parallel_world_size()

            if mp_size > 1 or sp_size > 1:
                torch.distributed.broadcast_object_list(broadcast_list, src=mpu.get_data_broadcast_src_rank(), group=mpu.get_data_broadcast_group())

            text, cnt, stopped = broadcast_list
            if stopped:
                break

            if mpu.get_data_broadcast_rank() == 0:
                print(cnt, ': ', text)

            if args.image2video:    # i2v 输入是图片+prompt
                if args.use_pose:
                    text_parts = text.split('@@')
                    text = text_parts[0]
                    input_dir = text_parts[1]
                    
                    # Find reference image with multiple possible names
                    ref_image_patterns = ['ref.jpg', 'ref.png', 'ref_image.jpg', 'ref_image.png']
                    image_path = find_file_with_patterns(input_dir, ref_image_patterns)
                    if image_path is None:
                        raise FileNotFoundError(f"Reference image not found in {input_dir}. Tried: {ref_image_patterns}")
                    
                    # Find pose video with multiple possible names
                    pose_patterns = ['rendered_aligned.mp4', 'rendered.mp4', 'rendered_v2.mp4']
                    pose_path = find_file_with_patterns(input_dir, pose_patterns)
                    if pose_path is None:
                        raise FileNotFoundError(f"Pose video not found in {input_dir}. Tried: {pose_patterns}")
                    
                    if "smpl_downsample_mask" in args.representation:
                        # Try replace_mask.mp4 first (no reversal needed), fallback to rendered_mask_v2.mp4
                        replace_mask_video_path = find_file_with_patterns(input_dir, ['replace_mask.mp4'])
                        if replace_mask_video_path is not None:
                            print(f"Using replace_mask.mp4 as mask video for {input_dir}")
                            mask_path = replace_mask_video_path
                            ref_mask_flag = False
                        else:
                            pose_mask_patterns = ['rendered_mask_v2.mp4']
                            mask_path = find_file_with_patterns(input_dir, pose_mask_patterns)
                            if mask_path is None:
                                raise FileNotFoundError(f"Mask video not found in {input_dir}. Tried: ['replace_mask.mp4', 'rendered_mask_v2.mp4']")
                            ref_mask_flag = True

                        ref_mask_path = find_file_with_patterns(input_dir, ['ref_mask.jpg', 'ref_mask.png'])
                        if ref_mask_path is None:
                            raise FileNotFoundError(f"Reference mask not found in {input_dir}. Tried: ['ref_mask.jpg']")

                    if text == "None":
                        text = ""
                    else:
                        text = text
                else:
                    text, image_path = text.split('@@')
                
                
                # ******获取动作序列******
                GT = None
                GT_patterns = ['GT.mp4']
                GT_path = find_file_with_patterns(input_dir, GT_patterns)
                if GT_path is not None:
                    GT = load_video_for_pose_sample(GT_path)
                    GT = GT.permute(0, 3, 1, 2) #
                    GT = (GT - 127.5) / 127.5   # color value: 0-255 -> -1-1

                if image_path != "firstframe":     
                    assert os.path.exists(image_path), "video should exist"
                    image_tensor = load_image_to_tensor_chw_normalized(image_path)
                else:                   # "firstframe" tag is for testing self-driven cases, directly using first frame of GT as reference image
                    assert GT is not None
                    image_tensor = GT[0].unsqueeze(0)    # C H W -> T C H W
                # 获取采样尺寸
                if image_tensor.shape[2] < image_tensor.shape[3]:
                    target_H, target_W = args.sampling_image_size
                else:
                    target_W, target_H = args.sampling_image_size

                
                # 获取驱动信号
                # Get fps from driving video
                decord.bridge.set_bridge("torch")
                vr_for_fps = VideoReader(uri=pose_path, height=-1, width=-1)
                driving_fps = vr_for_fps.get_avg_fps()
                print(f"Driving video fps: {driving_fps}")
                
                smpl_render_video = load_video_for_pose_sample(pose_path)
                smpl_render_video = smpl_render_video.permute(0, 3, 1, 2) # T H W C -> T C H W
                smpl_render_video = resize_for_rectangle_crop(smpl_render_video, [target_H, target_W], reshape_mode="center")
                smpl_render_video = (smpl_render_video - 127.5) / 127.5   # color value: 0-255 -> -1-1

                if "smpl_downsample" in args.representation:
                    smpl_render_video = F.interpolate(smpl_render_video, scale_factor=0.5, mode='bilinear', align_corners=False)  # t c h w
                    if "smpl_downsample_mask" in args.representation:
                        pose_sam_video = load_video_for_pose_sample(mask_path)
                        pose_sam_video = pose_sam_video.permute(0, 3, 1, 2) # T H W C -> T C H W
                        pose_sam_video = resize_for_rectangle_crop(pose_sam_video, [target_H, target_W], reshape_mode="center")
                        pose_sam_video = (pose_sam_video - 127.5) / 127.5   # color value: 0-255 -> -1-1
                        pose_sam_video = F.interpolate(pose_sam_video, scale_factor=0.5, mode='bilinear', align_corners=False)  # 0.5x, same as smpl_downsample
                        ref_sam = load_image_to_tensor_chw_normalized(ref_mask_path)
                        ref_sam = resize_for_rectangle_crop(ref_sam, [target_H, target_W], reshape_mode="center")  # 1 c h w, -1-1
                    else:
                        pose_sam_video = None
                        ref_sam = None
                sampling_num_frames = smpl_render_video.shape[0]

                # 其它的也都crop
                image_tensor = resize_for_rectangle_crop(image_tensor, [target_H, target_W], reshape_mode="center")
                if GT is not None:
                    GT = resize_for_rectangle_crop(GT, [target_H, target_W], reshape_mode="center")


                smpl_segments = []
                sam_segments = []
                if sampling_num_frames <= 81:
                    # 短视频，不需要分段；截到 4k+1 以匹配 Wan VAE 和 mask 压缩函数
                    T_keep = ((sampling_num_frames - 1) // 4) * 4 + 1
                    smpl_segments.append(smpl_render_video[:T_keep])
                    sam_segments.append(pose_sam_video[:T_keep] if pose_sam_video is not None else None)
                else:
                    # 长视频，分段处理
                    # 第一段：81帧
                    remaining_start = 0
                    # 后续段：每段81帧（包括5帧overlap用于历史）
                    while remaining_start < sampling_num_frames:
                        segment_end = remaining_start + 81
                        if segment_end >= sampling_num_frames:  # 不处理了
                            break
                        else:
                            smpl_segments.append(smpl_render_video[remaining_start:segment_end])
                            sam_segments.append(pose_sam_video[remaining_start:segment_end] if pose_sam_video is not None else None)
                            remaining_start += 76 # 76 77 78 79 80 -> history


                if "smpl_downsample_mask" in args.representation and ref_sam is not None:
                    # ref_sam: (1, C=3, H, W)，不做预下采样，additional_spatial_downsample=1
                    _ref_sam_28ch = extract_and_compress_mask_to_latent(
                        ref_sam.permute(1, 0, 2, 3).to('cuda'), additional_spatial_downsample=1
                    )  # (28, 1, H_lat, W_lat)
                    ref_sam_latent = _ref_sam_28ch.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # (1, 1, 28, H_lat, W_lat) = B T C H W
                else:
                    ref_sam_latent = None

                output_segments = []     # 如果是长视频，可能产生多个片段；仅 SP rank 0 上有内容
                prev_history_pixel = None  # 上一段末5帧 pixel space；所有 SP rank 都要持有，作为下一段 VAE encode 的输入
                for seg_idx, smpl_render_segment, sam_segment in zip(range(len(smpl_segments)), smpl_segments, sam_segments):
                    print("Processing segment %d / %d" % (seg_idx+1, len(smpl_segments)))
                    # VAE编码
                    if model.i2v_encode_video:          # wan的模式,不需要再重复或者替换第一帧
                        assert args.use_pose, 'wan for not using pose has not been merged into this version'
                        smpl_render_segment = smpl_render_segment.unsqueeze(0).to('cuda').to(torch.bfloat16)  # B T C H W
                        ori_image = image_tensor.unsqueeze(0).to('cuda').to(torch.bfloat16)  # B 1 C H W, -1-1
                        image_to_save = ori_image.repeat(1, smpl_render_segment.shape[1], 1, 1, 1)
                        image = torch.concat([ori_image, (torch.zeros_like(ori_image)).repeat(1, smpl_render_segment.shape[1] - 1, 1, 1, 1)], dim=1)
                        image = rearrange(image, 'b t c h w -> b c t h w').contiguous()
                        image = model.encode_first_stage(image, None, force_encode=True)
                        image = image.permute(0, 2, 1, 3, 4).contiguous() # BCTHW -> BTCHW
                        ref_concat = model.encode_first_stage(rearrange(ori_image, 'b t c h w -> b c t h w').contiguous() , None, force_encode=True)
                        ref_concat = ref_concat.permute(0, 2, 1, 3, 4).contiguous()
                    else:                               # 旧的cogvideo的模式，如果用到需要重写
                        raise NotImplementedError("Old cogvideo i2v encoding not implemented yet")


                    assert "smpl" in args.representation
                    smpl_render_latent = model.encode_first_stage(rearrange(smpl_render_segment, 'b t c h w -> b c t h w').contiguous(), None, force_encode=True)
                    smpl_render_latent = smpl_render_latent.permute(0, 2, 1, 3, 4).contiguous()   # B, T, C, H, W
                    use_null_pose = False  # set True to replace pose with null (zero) pose, consistent with training pose dropout
                    if use_null_pose:
                        T_l, H_l, W_l = smpl_render_latent.shape[1], smpl_render_latent.shape[3], smpl_render_latent.shape[4]
                        null_smpl = torch.load(f"latents/zero_pose_latent_{T_l}_{H_l}_{W_l}.pt").to('cuda').to(torch.bfloat16)
                        smpl_render_latent = null_smpl.unsqueeze(0).expand_as(smpl_render_latent).contiguous()
                    if "smpl_downsample_mask" in args.representation:
                        # sam_segment: (T, C=3, H/2, W/2)，已 2x 预下采样，additional_spatial_downsample=1
                        _sam_28ch = extract_and_compress_mask_to_latent(
                            sam_segment.permute(1, 0, 2, 3).to('cuda'), additional_spatial_downsample=1
                        )  # (28, T_lat, H_lat, W_lat)
                        sam_latent = _sam_28ch.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous()  # B T C H W

                    T = smpl_render_latent.shape[1]
                    C, H, W = image.shape[2], image.shape[3], image.shape[4]


                    # 处理历史帧
                    history_latent = None
                    history_mask = None
                    if seg_idx > 0:
                        # 有历史帧，需要VAE编码
                        assert prev_history_pixel is not None, "prev_history_pixel 未填充：多段采样且 only_save_latents=True 时不支持"
                        print(f"Using history frames from previous segment")
                        print(f"History frames shape: {prev_history_pixel.shape}")
                        history_frames = prev_history_pixel  # B 5 C H W, 5帧历史
                        history_frames = history_frames.to('cuda').to(torch.bfloat16)  # B T C H W
                        history_latent = model.encode_first_stage(rearrange(history_frames, 'b t c h w -> b c t h w').contiguous(), None, force_encode=True)
                        history_latent = history_latent.permute(0, 2, 1, 3, 4).contiguous()  # B T C H W
                        
                        # 创建history_mask: b t 4 h w，前2帧为1
                        history_mask = torch.zeros(1, T, 4, H, W).to('cuda').to(torch.bfloat16)
                        # 假设历史帧对应前2个latent帧（5个pixel帧 -> 2个latent帧）
                        history_mask[:, :2, :, :, :] = 1
                        print(f"History latent shape: {history_latent.shape}, mask shape: {history_mask.shape}")


                    if model.use_i2v_clip:
                        model.i2v_clip.model.to('cuda')
                        image_clip_features = model.i2v_clip.visual(ori_image.permute(0, 2, 1, 3, 4))  # btchw -> bcthw
                        model.i2v_clip.model.cpu()
                    

                    # TODO: broadcast image2video
                    value_dict = {
                        'prompt': text,
                        # 'negative_prompt': "手部变形，脸部变形，低质量",
                        'negative_prompt': "",
                        'num_frames': torch.tensor(T).unsqueeze(0)
                    }
                    test_case_idx = os.path.basename(input_dir)  
                    save_dir = os.path.join(args.output_dir, test_case_idx)
                    os.makedirs(save_dir, exist_ok=True)
                    with open(os.path.join(save_dir, 'text.txt'), 'w') as f:
                        f.write(text)

                    model.conditioner.embedders[0].to('cuda')
                    batch, batch_uc = get_batch(
                        get_unique_embedder_keys_from_conditioner(model.conditioner),
                        value_dict,
                        num_samples
                    )
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            print(key, batch[key].shape)
                        elif isinstance(batch[key], list):
                            print(key, [len(l) for l in batch[key]])
                        else:
                            print(key, batch[key])
                    
                    # 这里把batch加上text embedding包装成c和uc
                    c, uc = model.conditioner.get_unconditional_conditioning(
                        batch,
                        batch_uc=batch_uc,
                        force_uc_zero_embeddings=force_uc_zero_embeddings,
                    )
                    model.conditioner.embedders[0].cpu()

                    for k in c:
                        if not k == "crossattn":
                            c[k], uc[k] = map(
                                lambda y: y[k][: math.prod(num_samples)].to("cuda"), (c, uc)
                            )

                    if args.image2video:
                        assert not args.multi_cond_cfg, "Multi Cond CFG does not work well"
                        if args.use_pose:
                            c["concat_images"] = image
                            uc["concat_images"] = image
                            c["ref_concat"] = ref_concat
                            uc["ref_concat"] = ref_concat
                            if seg_idx > 0:
                                c["history"] = history_latent
                                uc["history"] = history_latent
                                c["history_mask"] = history_mask
                                uc["history_mask"] = history_mask
                            if "smpl" in args.representation:
                                c["concat_smpl_render"] = smpl_render_latent
                                uc["concat_smpl_render"] = smpl_render_latent
                                if "smpl_downsample_mask" in args.representation:
                                    c["concat_latent_sam"] = sam_latent
                                    uc["concat_latent_sam"] = sam_latent
                                    T_l = smpl_render_latent.shape[1]
                                    null_noisy_mask = torch.zeros(T_l, ref_sam_latent.shape[2], ref_sam_latent.shape[3], ref_sam_latent.shape[4], device='cuda', dtype=torch.bfloat16)  # (T_smpl, C=28, H, W)
                                    null_noisy_mask = null_noisy_mask.unsqueeze(0).expand(ref_sam_latent.shape[0], -1, -1, -1, -1).contiguous()
                                    ref_sam_latent_full = torch.cat([ref_sam_latent, null_noisy_mask], dim=1)
                                    c["concat_latent_ref_mask"] = ref_sam_latent_full
                                    uc["concat_latent_ref_mask"] = ref_sam_latent_full
                                    c["ref_mask_flag"] = torch.tensor([ref_mask_flag], dtype=torch.bool).to('cuda')
                                    uc["ref_mask_flag"] = torch.tensor([ref_mask_flag], dtype=torch.bool).to('cuda')
                        else:
                            c["concat_images"] = image     # torch.Size([1, 32, 128, 32, 55])
                            uc["concat_images"] = image     # 如果为zeros_like t2v结果也不变
                        if model.use_i2v_clip:
                            c["image_clip_features"] = image_clip_features
                            uc["image_clip_features"] = image_clip_features

                            


                    assert args.batch_size == 1, "Only batch size 1 is supported in long video inference"
                    if args.multi_cond_cfg:
                        raise NotImplementedError("Multi Cond CFG does't work well")
                    else:
                        samples_z = sample_func(
                            c,
                            uc = uc,
                            batch_size = 1,
                            shape = (T, C, H, W),
                            ofs = torch.tensor([2.0]).to('cuda'),
                            fps = torch.tensor([args.sampling_fps]).to('cuda'),
                        )
                    if mpu.get_sequence_parallel_rank() == 0:
                        samples_z = samples_z.permute(0, 2, 1, 3, 4).contiguous()
                        if args.only_save_latents:
                            if mpu.get_model_parallel_rank() == 0:
                                samples_z = 1.0 / model.scale_factor * samples_z
                                # torch.save(samples_z, save_path)
                        else:
                            samples_x = model.decode_first_stage(samples_z).to(torch.float32)
                            # samples_x = samples_x.view(b, t, *samples_x.shape[1:])
                            samples_x = samples_x.permute(0, 2, 1, 3, 4).contiguous() 
                            if seg_idx == 0:
                                # 第一段，全部保留
                                last_latent = samples_x.cpu()  # b t c h w
                            else:
                                # 后续段去掉前5帧重叠（history已锚到上一段末尾）
                                last_latent = samples_x.cpu()[:, 5:]  # b t c h w
                            # samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0).cpu()  # b t c h w

                        output_segments.append(last_latent)

                    # 准备下一段的 5 帧历史 pixel：从 SP rank 0 广播到全部 SP rank（每个 rank 仍按现有模式独立 VAE encode）
                    if not args.only_save_latents and seg_idx < len(smpl_segments) - 1:
                        sp_size = mpu.get_sequence_parallel_world_size()
                        if mpu.get_sequence_parallel_rank() == 0:
                            next_history = samples_x[:, -5:].contiguous().to('cuda').to(torch.bfloat16)
                        else:
                            next_history = torch.empty(
                                1, 5, 3, target_H, target_W, device='cuda', dtype=torch.bfloat16,
                            )
                        if sp_size > 1:
                            torch.distributed.broadcast(
                                next_history,
                                src=mpu.get_sequence_parallel_src_rank(),
                                group=mpu.get_sequence_parallel_group(),
                            )
                        prev_history_pixel = next_history



            if mpu.get_sequence_parallel_rank() == 0:
                final_samples = torch.cat(output_segments, dim=1)
                final_samples = torch.clamp((final_samples + 1.0) / 2.0, min=0.0, max=1.0)
                if mpu.get_model_parallel_rank() == 0:
                    save_multi_video_grid_and_mp4([final_samples], save_dir, fps=driving_fps, key=f"{test_case_idx}_output")

                    # Save GT (top) + generated video (bottom) vertically concatenated
                    if GT is not None:
                        B, T_out, C_out, H_out, W_out = final_samples.shape
                        gt_for_vis = torch.clamp((GT + 1.0) / 2.0, 0.0, 1.0)  # T C H W
                        T_gt = min(gt_for_vis.shape[0], T_out)
                        gt_for_vis = F.interpolate(gt_for_vis[:T_gt], size=(H_out, W_out), mode='bilinear', align_corners=False)
                        vid_for_vis = final_samples[0, :T_gt]  # T C H W, 0~1
                        combined_frames = []
                        for t in range(T_gt):
                            gt_frame = (255.0 * rearrange(gt_for_vis[t], "c h w -> h w c").cpu().numpy()).astype(np.uint8)
                            vid_frame = (255.0 * rearrange(vid_for_vis[t], "c h w -> h w c").cpu().numpy()).astype(np.uint8)
                            combined_frames.append(np.concatenate([gt_frame, vid_frame], axis=0))
                        gt_overlay_path = os.path.join(save_dir, f"{test_case_idx}_gt_overlay.mp4")
                        with imageio.get_writer(gt_overlay_path, fps=driving_fps) as writer:
                            for frame in combined_frames:
                                writer.append_data(frame)
                

if __name__ == '__main__':
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
        os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
        os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    del args.deepspeed_config
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    if "sigma_sampler_config" in args.model_config.loss_fn_config.params.keys() and hasattr(args.model_config.loss_fn_config.params.sigma_sampler_config.params, "uniform_sampling"):
        args.model_config.loss_fn_config.params.sigma_sampler_config.params.uniform_sampling = False

    if args.model_type == "dit":
        Engine = diffusion_video.SATVideoDiffusionEngine
    else:
        raise NotImplementedError(f"model_type={args.model_type!r} is not supported in this inference release")
    print(args.model_type)

    sampling_main(args, model_cls=Engine)
