import io
import re
import os
import sys
import numpy as np
from PIL import Image
from functools import partial
import math
import torch

import torchvision.transforms as TT
from torch.utils.data import default_collate

from sgm.webds import MetaDistributedWebDataset
from webdataset import DataPipeline

import random
import numpy as np
import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode

from sgm.util import instantiate_from_config
from sat.helpers import print_rank0
import decord
from decord import VideoReader
import imageio

def rectangle_crop(arr, image_size, reshape_mode='center'):
    h, w = arr.shape[2], arr.shape[3]
    new_h, new_w = image_size

    delta_h = h - new_h
    delta_w = w - new_w

    if reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(
        arr, top=top, left=left, height=new_h, width=new_w
    )
    return arr



def resize_for_rectangle_crop(arr, image_size, reshape_mode='random'):
    if arr.shape[3] / arr.shape[2] > image_size[1] / image_size[0]:
        arr = resize(
            arr,
            size=[image_size[0], int(arr.shape[3] * image_size[0] / arr.shape[2])],
            interpolation=InterpolationMode.BICUBIC,
        )
    else:
        arr = resize(
            arr,
            size=[int(arr.shape[2] * image_size[1] / arr.shape[3]), image_size[1]],
            interpolation=InterpolationMode.BICUBIC,
        )

    h, w = arr.shape[2], arr.shape[3]

    delta_h = h - image_size[0]
    delta_w = w - image_size[1]

    if reshape_mode == "random" or reshape_mode == "none":
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
    elif reshape_mode == "center":
        top, left = delta_h // 2, delta_w // 2
    else:
        raise NotImplementedError
    arr = TT.functional.crop(
        arr, top=top, left=left, height=image_size[0], width=image_size[1]
    )
    return arr


def pad_last_frame(tensor, sampling_frms_num):
    # T, H, W, C
    if tensor.shape[0] < sampling_frms_num:
        # 复制最后一帧
        last_frame = tensor[-int(sampling_frms_num - tensor.shape[0]) :]
        # 将最后一帧添加到第二个维度
        padded_tensor = torch.cat([tensor, last_frame], dim=0)
        return padded_tensor
    else:
        return tensor[:sampling_frms_num]


def load_video(
    video_data,
    sampling="uniform",
    duration=None,
    num_frames=4,
    wanted_fps=None,
    actual_fps=None,
    skip_frms_num=0.0,
    ori_height=None,
    ori_width=None,
    image_size=None,
):
    # num_frames: wanted frames in wanted fps; image_size: [H, W]
    if ori_width / ori_height > image_size[1] / image_size[0]:
        new_height = image_size[0]
        new_width = int(ori_width * new_height / ori_height)
    else:
        new_width = image_size[1]
        new_height = int(ori_height * new_width / ori_width)

    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_data, height=new_height, width=new_width)
    ori_vlen = min(int(duration * actual_fps) - 1, len(vr))

    start = skip_frms_num
    end = int(start + num_frames / wanted_fps * actual_fps)

    if sampling == "uniform":
        indices = np.arange(start, end, (end - start) / num_frames).astype(int)
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(np.arange(0, end))
    assert temp_frms is not None
    tensor_frms = (
        torch.from_numpy(temp_frms)
        if type(temp_frms) is not torch.Tensor
        else temp_frms
    )
    tensor_frms = tensor_frms[indices.tolist()]

    return pad_last_frame(tensor_frms, num_frames)

import threading
import ctypes
import inspect

def _async_raise(tid, exctype):
    """Raises an exception in the threads with id tid"""
    if not inspect.isclass(exctype):
        raise TypeError("Only types can be raised (not instances)")
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")

def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)

def load_video_with_timeout(*args, **kwargs):
    # 创建一个Thread对象，目标函数是load_video
    video_container = {}
    def target_function():
        video = load_video(*args, **kwargs)
        video_container['video'] = video

    # 启动线程
    thread = threading.Thread(target=target_function)
    thread.start()
    # 等待线程完成或超时
    timeout = 10
    # timeout = 30
    thread.join(timeout)
    if thread.is_alive():
        # stop_thread(thread)
        # thread.join()
        print("Loading video timed out")
        raise TimeoutError
        # return None  # 可以抛出异常或返回特定值表示超时
    return video_container.get('video', None).contiguous()


def process_video(video_path, image_size=None, duration=None, num_frames=4, wanted_fps=None, actual_fps=None, skip_frms_num=0., reader_type='decord',
                  ori_height=None, ori_width=None):
    '''
        video_path: str or io.BytesIO
        image_size: .
        duration: preknow the duration to speed up by seeking to sampled start. TODO by_pass if unknown.
        num_frames: wanted num_frames.
        wanted_fps: .
        skip_frms_num: ignore the first and the last xx frames, avoiding transitions.
    '''

    assert reader_type == 'decord'
    video = load_video_with_timeout(video_path, duration=duration, num_frames=num_frames, wanted_fps=wanted_fps, image_size=image_size,
                           actual_fps=actual_fps, skip_frms_num=skip_frms_num, ori_height=ori_height, ori_width=ori_width)

    # --- copy and modify the image process ---
    video = video.permute(0, 3, 1, 2) # [T, C, H, W]

    # resize
    if image_size is not None:
        # video = resize(video, image_size, interpolation=InterpolationMode.BICUBIC)
        # video = resize_for_rectangle_crop(video, image_size, reshape_mode="center")
        video = rectangle_crop(video, image_size)

    return video.contiguous()


class VideoWebDataset(MetaDistributedWebDataset):
    def __init__(
        self,
        path,
        fps,
        num_frames=None,
        image_size=None,
        filters=None,
        meta_names=None,
        reader_type='decord',
        extra_texts=None,
        skip_frms_num=0.,
        nshards=sys.maxsize,
        modify_prompt=False,
        add_stock_prefix=True,
        seed=-1,
        shuffle_buffer=16,
        include_dirs=None,
        bucket_helper=None,
        valid=False,
        use_cluster=False,
        **extra_kwarg
    ):
        self.path = path
        if seed == -1:
            seed = random.randint(0, 1000000)
        if meta_names is None:
            meta_names = []
        if path.startswith(';'):
            path, include_dirs = path.split(';', 1)

        txt_keys, cum_sum, c_sum = [], [], 0
        self.extra_texts = extra_texts
        if extra_texts is None:
            self.extra_texts = []
        for extra_text_item in self.extra_texts:
            key, prob = extra_text_item['key'], extra_text_item['prob']

            txt_keys.append(key)
            c_sum += prob
            cum_sum.append(c_sum)

        txt_keys.append('txt')
        self.txt_keys = txt_keys
        self.cum_sum = np.array(cum_sum)

        self.fps = fps
        self.skip_frms_num = int(skip_frms_num)
        self.image_size = image_size
        self.num_frames = num_frames
        self.bucket_helper = bucket_helper
        self.modify_prompt = modify_prompt
        self.add_stock_prefix = add_stock_prefix
        self.reader_type = reader_type
        self.use_cluster = use_cluster

        if filters is None:
            filters = []

        self.filters = filters

        super().__init__(
            path,
            self.process,
            seed=seed,
            meta_names=meta_names,
            shuffle_buffer=shuffle_buffer,
            nshards=nshards,
            include_dirs=include_dirs
        )


    def process(self, src):
        for r in src:
            filter_flag = 0

            for f in self.filters:
                key = f['key']
                default_score = -float('inf') if f["greater"] else float('inf')
                score = r.get(key, default_score) or default_score
                judge = (lambda a: a > f["val"]) if f["greater"] else (lambda a: a < f["val"])
                if not judge(score):
                    filter_flag = 1
                    break
            if filter_flag:
                continue

            if self.use_cluster:
                if r.get('cluster_chosen_0', None) is not None and r['cluster_chosen_0'] == False:
                    # print_rank_0(f'skip cluster not chosen data: {r["__key__"]} of {r["__url__"]}')
                    continue

            # filter movie year
            if r.get('movie_year', None) is not None:
                if int(r['movie_year']) < 2000:
                    continue
                
            ofs = r.get('optical_flow_score', None)
            if ofs is not None:
                ofs = float(ofs)
            else:
                continue

            duration = r.get('duration', None)
            if duration is not None:
                duration = float(duration)
            else:
                continue

            if duration < 3.1 or duration > 30:
                continue

            actual_fps=r.get('fps', None)
            if actual_fps is not None:
                actual_fps = float(actual_fps)
            if actual_fps is None:
                continue

            # pick text
            idx = ((self.cum_sum - np.random.random()) < 0).sum()
            txt_key = self.txt_keys[idx]
            txt = r.get(txt_key, None)
            
            if txt is None and txt_key == 'caption':
                txt_key = 'long_caption_v3_96frame'
                txt = r.get(txt_key, None)
            if txt is None and txt_key == 'long_caption_v3_96frame':
                txt_key = 'long_caption_v3'
                txt = r.get(txt_key, None)
            if txt is None and txt_key == 'long_caption_v3':
                txt_key = 'long_captionv2'
                txt = r.get(txt_key, None)
            if txt is None and txt_key == 'long_captionv2':
                txt_key = 'long_caption'
                txt = r.get(txt_key, None)
            if txt is None and txt_key == 'recaption_en':
                txt_key = 'caption'
                txt = r.get(txt_key, None)
            if txt is None and txt_key == 'new_caption':
                txt_key = 'old_caption'
                txt = r.get(txt_key, None)
            if txt is None:
                if txt_key != 'caption':
                    txt_key = 'caption'
                else:
                    txt_key = 'long_caption_v3_96frame'
                txt = r.get(txt_key, None)

            if txt is None or txt == 'None':
                continue

            # 去掉固定前缀句式
            if self.modify_prompt:
                words = txt.split()
                if len(words) < 5:
                    continue
                if words[1].lower() == 'video' or words[1].lower() == 'image':
                    words[3] = words[3].capitalize()
                    txt = ' '.join(words[3:])
            
            # sucai
            if self.add_stock_prefix:
                if 'pond5' in r['__url__'] or 'elements' in r['__url__'] or 'stock' in r['__url__'] or 'webvid' in r['__url__']:
                    txt = 'Stock video. ' + txt

            if 'mp4' in r:
                video_data = r['mp4']
            elif 'avi' in r:
                video_data = r['avi']
            else:
                print('No video data found')
                continue

            h = r.get('height', None)
            w = r.get('width', None)
            if h is None or w is None:
                continue

            h = int(h)
            w = int(w)
            wanted_fps = self.fps
            avail_frames_in_actual_fps = int(duration * actual_fps) - self.skip_frms_num * 2
            avail_frames_in_wanted_fps = int(duration * wanted_fps - self.skip_frms_num * 2 * wanted_fps / actual_fps)

            if self.bucket_helper is not None:
                bucket_id, bucket_shape = self.bucket_helper.assign_bucket(avail_frames_in_wanted_fps, h, w, is_image=False)
                if bucket_id is None or bucket_shape is None:
                    continue
            else:
                bucket_id = -1
                bucket_shape = (self.num_frames,) + self.image_size

            bucket_t, bucket_h, bucket_w = bucket_shape

            try:
                frames = process_video(io.BytesIO(video_data), num_frames=bucket_t, wanted_fps=wanted_fps, image_size=[bucket_h, bucket_w], duration=duration, \
                                       actual_fps=actual_fps, skip_frms_num=self.skip_frms_num, ori_height=h, ori_width=w)
                frames = (frames - 127.5) / 127.5

            except Exception as e:
                print(e, self.path)
                continue

            item = {
                'mp4': frames, # TCHW
                'txt': txt,
                'num_frames': bucket_t,
                'fps': self.fps,
                'bucket_id': bucket_id
            }
            yield item


class ImageWebDataset(MetaDistributedWebDataset):
    def __init__(
        self,
        path,
        interpolation=None,
        image_size=None,
        nshards=sys.maxsize,
        seed=-1,
        meta_names=None,
        shuffle_buffer=16,
        include_dirs=None,
        shape_filter='bigger',
        filters=None,
        extra_texts=None,
        modify_prompt=False,
        reshape_mode='center',
        bucket_helper=None,
        **extra_kwargs
    ):
        self.path = path
        if seed == -1:
            seed = random.randint(0, 1000000)
        if meta_names is None:
            meta_names = []
        if path.startswith(';'):
            path, include_dirs = path.split(';', 1)

        chained_transforms = []
        # if reshape_mode != 'none':
        #     chained_transforms.append(TT.Resize(size=image_size[1], interpolation=interpolation))
        chained_transforms.append(TT.ToTensor())
        chained_transforms = TT.Compose(chained_transforms)
        self.transform = chained_transforms
        self.image_size = image_size
        self.extra_texts = extra_texts
        self.modify_prompt = modify_prompt
        self.reshape_mode = reshape_mode
        self.shape_filter = shape_filter
        self.bucket_helper = bucket_helper

        if filters is None:
            filters = []
        self.filters = filters

        txt_keys, cum_sum, c_sum = [], [], 0
        if self.extra_texts is None:
            self.extra_texts = []
        for extra_text_item in self.extra_texts:
            key, prob = extra_text_item['key'], extra_text_item['prob']

            txt_keys.append(key)
            c_sum += prob
            cum_sum.append(c_sum)
        txt_keys.append('txt')
        self.txt_keys = txt_keys
        self.cum_sum = np.array(cum_sum)

        super().__init__(
            path,
            self.process,
            seed,
            meta_names=meta_names,
            shuffle_buffer=shuffle_buffer,
            nshards=nshards,
            include_dirs=include_dirs
        )

    def process(self, src):
        for r in src:
            # read Image
            if ('png' not in r and 'jpg' not in r):
                continue

            filter_flag = 0
            for f in self.filters:
                key = f['key']
                default_score = -float('inf') if f["greater"] else float('inf')
                score = r.get(key, default_score) or default_score
                judge = (lambda a: a > f["val"]) if f["greater"] else (lambda a: a < f["val"])
                if not judge(score):
                    filter_flag = 1
                    break
            if filter_flag:
                continue

            img_bytes = r['png'] if 'png' in r else r['jpg']
            try:
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            except Exception as e:
                print(e, self.path)
                continue

            w, h = img.size

            arr = self.transform(img) # ToTensor
            arr = arr.unsqueeze(0) # TCHW

            if self.bucket_helper is not None:
                bucket_id, bucket_shape = self.bucket_helper.assign_bucket(1, h, w, is_image=True)
                if bucket_id is None or bucket_shape is None:
                    continue
            else:
                bucket_id = -1
                bucket_shape = (1,) + self.image_size

            bucket_t, bucket_h, bucket_w = bucket_shape

            if self.shape_filter == 'all':
                pass

            elif self.shape_filter == 'bigger':
                if w <= bucket_w or h <= bucket_h:
                    continue

            else:
                raise NotImplementedError

            arr = resize_for_rectangle_crop(arr, (bucket_h, bucket_w), reshape_mode=self.reshape_mode) # TCHW
            arr = arr * 2 - 1

            # pick text
            idx = ((self.cum_sum - np.random.random()) < 0).sum()
            txt_key = self.txt_keys[idx]
            if 'datacomp' in r['__url__']:
                if txt_key == 'recaption_en':
                    txt_key = 'new_caption'
                elif txt_key == 'caption':
                    txt_key = 'old_caption'
            
            txt = r.get(txt_key, None)
            if txt is None or txt == 'None':
                continue

            # 去掉固定前缀句式
            if self.modify_prompt:
                words = txt.split()
                if len(words) < 5:
                    continue
                if words[1].lower() == 'video' or words[1].lower() == 'image':
                    words[3] = words[3].capitalize()
                    txt = ' '.join(words[3:])

            item = {
                'mp4': arr, # TCHW
                'txt': txt,
                'num_frames': 1,
                'fps': 0,
                'bucket_id': bucket_id
            }
            yield item


import zstandard as zstd
def deserialize_tensor_from_zstd(buffer):
    """从 Zstandard 压缩格式的字节反序列化为 PyTorch 张量."""
    dctx = zstd.ZstdDecompressor()
    decompressed = dctx.decompress(buffer)
    buffer = io.BytesIO(decompressed)
    tensor = torch.load(buffer, map_location='cpu')
    return tensor


def process_fn_video_latent(src, image_size, frame_range, fps, inner_batch_size, filters, extra_texts, i2v=False, modify_prompt=False):
    while True:
        choice_id = random.randint(0, len(frame_range) - 1)
        num_frames, num_videos = frame_range[choice_id]
        num_videos = num_videos * inner_batch_size
        now_num_videos = 0
        txt_list = []
        sampled_frames = []
        first_frame_list = []
        txt_keys, cum_sum, c_sum = [], [], 0
        for extra_text_item in extra_texts:
            key, prob = extra_text_item['key'], extra_text_item['prob']

            txt_keys.append(key)
            c_sum += prob
            cum_sum.append(c_sum)
        txt_keys.append('caption')
        cum_sum = np.array(cum_sum)

        while now_num_videos < num_videos:
            r = next(src)

            # if r['movie_year'] is not None:
            #     if float(r['movie_year']) <= 2005:
            #         continue
            # if r['movie_language'] is not None:
            #     if r['movie_language'] not in ['zh', 'cn', 'ja', 'ko','vi' ,'th' ,'cmn', 'my' ,'yue' ,'lo']:
            #         continue

            filter_flag = 0
            if filters is None:
                filters = []
            for filter in filters:
                key = filter['key']
                default_score = -float('inf') if filter["greater"] else float('inf')
                score = r.get(key, default_score) or default_score
                judge = (lambda a: a > filter["val"]) if filter["greater"] else (lambda a: a < filter["val"])
                if not judge(score):
                    filter_flag = 1
                    break
            if filter_flag:
                continue

            # if 'width' in r:
            #     width = r['width']
            #     height = r['height']
            #     if width < image_size[1] or height < image_size[0]:
            #         print('Image size too small', width, height, image_size[1], image_size[0])
            #         continue
            # pick text
            idx = ((cum_sum - np.random.random()) < 0).sum()
            txt_key = txt_keys[idx]
            txt = r[txt_keys[idx]]
            if txt is None and txt_key == 'long_captionv2':
                txt_key = 'long_caption'
                txt = r[txt_key]
            if txt is None:
                txt_key = 'caption'
                txt = r[txt_key]
            if txt is None:
                continue
            if isinstance(txt, bytes):
                txt = txt.decode('utf-8')
            else:
                txt = str(txt)
            if txt.startswith('GeneratedText'):
                start_idx = len('GeneratedText(text="')
                txt = txt[start_idx:]
                end_idx = txt.find('generated_tokens=')
                txt = txt[:end_idx-3]
            if txt == 'None':
                continue
            if modify_prompt and txt_key == 'long_caption':
                words = txt.split()
                if len(words) < 5:
                    continue
                if words[1].lower() == 'video':
                    words[3] = words[3].capitalize()
                    txt = ' '.join(words[3:])


            frame = r.get('frame', None)
            if frame is not None:
                frame = int(frame)
            else:
                continue

            if now_num_videos == 0:
                while choice_id < len(frame_range) - 1 and frame < num_frames:
                    choice_id += 1
                    num_frames, num_videos = frame_range[choice_id]
                    num_videos = num_videos * inner_batch_size

            if frame < num_frames:
                if choice_id < len(frame_range) - 1:
                    choice_id += 1
                    num_frames, num_videos = frame_range[choice_id]
                    num_videos = num_videos * inner_batch_size
                    now_num_videos = 0
                    sampled_frames = []
                    txt_list = []
                continue


            if 'pth' in r.keys():
                try:
                    frames = torch.load(io.BytesIO(r['pth']), map_location='cpu')
                    frames = frames[:, :num_frames]
                    frames = frames.float()
                except Exception as e:
                    print(e)
                    continue
            elif 'pth.zstd' in r.keys():
                try:
                    buffer = r["pth.zstd"]
                    frames = deserialize_tensor_from_zstd(buffer).float()
                    frames = frames[:, :num_frames]
                except Exception as e:
                    print(e)
                    continue

            if i2v:
                try:
                    first_frame_buffer = r["first_frame_pth.zstd"]
                    first_frame = deserialize_tensor_from_zstd(first_frame_buffer).float()
                    frames = torch.cat([first_frame, frames], dim=1)
                except Exception as e:
                    print(e)
                    continue

            txt_list.append(txt)
            sampled_frames.append(frames)
            now_num_videos += 1

        sampled_frames = torch.stack(sampled_frames, dim=0) # (b c t h w)
        item = {
            'mp4': sampled_frames,
            'txt': txt_list,
            'num_frames': torch.tensor(num_frames),
            'fps': torch.tensor(fps),
        }

        yield item

def log_video_test(video_tensor, key, fps, log_video_test_dir = "log_video"):
    """
    :param video_tensor: torch.Tensor (T, C, H, W), 值范围[-1,1]
    :param pose_tensor: torch.Tensor (T, C, H, W), 值范围[-1,1]
    :param key: str
    :param fps: 保存视频的帧率
    """
    os.makedirs(log_video_test_dir, exist_ok=True)

    target_video_path = os.path.join(log_video_test_dir, key + '.mp4')

    def tensor_to_video(tensor, save_path):
        frames = []
        for frame in tensor:
            # (C, H, W) -> (H, W, C)
            frame = frame.permute(1, 2, 0)
            # [-1,1] -> [0,255]
            frame = ((frame + 1) / 2 * 255.0).clamp(0, 255)
            frame = frame.cpu().numpy().astype(np.uint8)
            frames.append(frame)

        with imageio.get_writer(save_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)

    tensor_to_video(video_tensor, target_video_path)


def random_between(a, b):
    return random.randint(min(a, b), max(a, b))


def get_zeros_like_compressed_frames(frames, temporal_compression_stride=4):
    # 计算VAE压缩后的shape
    T, _, H, W = frames.shape
    T_latent = (T - 1) // temporal_compression_stride + 1  # 一定是4n+1
    # WanVAE使用3次2倍下采样，每次下采样时如果尺寸是奇数会进行padding
    # 所以需要模拟实际的下采样过程
    H_temp = H
    W_temp = W
    for _ in range(3):  # 3次下采样
        H_temp = (H_temp + 1) // 2  # 模拟padding后的下采样
        W_temp = (W_temp + 1) // 2
    C = 16  # vae模型输入的通道数
    H_latent = H_temp
    W_latent = W_temp
    
    # 创建VAE压缩后的latent tensor
    latent_zeros = torch.zeros(T_latent, C, H_latent, W_latent, device=frames.device, dtype=frames.dtype)
    return latent_zeros


def gen_latent_bbox_mask(zeros_like_compressed_frames, bboxes, temporal_compression_stride=4):
    """
    Args:
        frames (torch.Tensor): 原始视频帧，形状为 (T, C, H, W)
        bboxes (List[List[List[float]]]): 
            一个嵌套列表。外层列表的长度应等于原始视频的像素帧数（例如65）。内层列表包含该帧上所有的BBox。每个BBox是一个列表 `[x_min, y_min, x_max, y_max]`，其中坐标是相对位置（0-1之间）。例如: [[[0.1, 0.2, 0.5, 0.6]], [[0.3, 0.4, 0.7, 0.8], [0.1, 0.1, 0.2, 0.3]], ...]
    """
    latent_bbox_mask = zeros_like_compressed_frames
    T_latent, H_latent, W_latent = latent_bbox_mask.shape[0], latent_bbox_mask.shape[2], latent_bbox_mask.shape[3]
    if bboxes is None or len(bboxes) == 0:
        return latent_bbox_mask

    # Part 1: 处理首帧
    bboxes_for_frame_0 = bboxes[0]
    for bbox in bboxes_for_frame_0:
        x_min_rel, y_min_rel, x_max_rel, y_max_rel = bbox
        # 将相对坐标转换为latent坐标并确保在有效范围内
        x_min_l = max(0, min(int(x_min_rel * W_latent), W_latent - 1))
        y_min_l = max(0, min(int(y_min_rel * H_latent), H_latent - 1))
        x_max_l = max(x_min_l + 1, min(math.ceil(x_max_rel * W_latent), W_latent))
        y_max_l = max(y_min_l + 1, min(math.ceil(y_max_rel * H_latent), H_latent))
        latent_bbox_mask[0, :, y_min_l:y_max_l, x_min_l:x_max_l] = 1.0          # 绘制（自动实现并集）

    # Part 2: 处理后续序列 (Pixel Frames 1+ -> Latent Frames 1+)
    for n in range(1, T_latent):
        # 计算对应的pixel帧范围
        start_pixel_index = (n - 1) * temporal_compression_stride + 1   # 比如n=1, start_pixel_index=1, end_pixel_index=4
        end_pixel_index = n * temporal_compression_stride
        for i in range(start_pixel_index, end_pixel_index + 1):
            if i < len(bboxes):
                bboxes_for_frame_i = bboxes[i]
                for bbox in bboxes_for_frame_i:
                    x_min_rel, y_min_rel, x_max_rel, y_max_rel = bbox
                    # 将相对坐标转换为latent坐标并确保在有效范围内
                    x_min_l = max(0, min(int(x_min_rel * W_latent), W_latent - 1))
                    y_min_l = max(0, min(int(y_min_rel * H_latent), H_latent - 1))
                    x_max_l = max(x_min_l + 1, min(math.ceil(x_max_rel * W_latent), W_latent))
                    y_max_l = max(y_min_l + 1, min(math.ceil(y_max_rel * H_latent), H_latent))
                    latent_bbox_mask[n, :, y_min_l:y_max_l, x_min_l:x_max_l] = 1.0          # 绘制（自动实现并集）
    return latent_bbox_mask


class VideoPoseLatentDataset(MetaDistributedWebDataset):
    def __init__(
        self,
        path,
        fps,
        downsample=True,
        num_frames=None,
        image_size=None,
        filters=None,
        meta_names=None,
        reader_type='decord',
        extra_texts=None,
        skip_frms_num=0.,
        nshards=sys.maxsize,
        modify_prompt=False,
        add_stock_prefix=True,
        seed=-1,
        shuffle_buffer=16,
        include_dirs=None,
        bucket_helper=None,
        valid=False,
        use_cluster=False,
        **extra_kwarg
    ):
        self.path = path
        if seed == -1:
            seed = random.randint(0, 1000000)
        if meta_names is None:
            meta_names = []
        if path.startswith(';'):
            path, include_dirs = path.split(';', 1)

        txt_keys, cum_sum, c_sum = [], [], 0
        self.extra_texts = extra_texts
        if extra_texts is None:
            self.extra_texts = []
        for extra_text_item in self.extra_texts:
            key, prob = extra_text_item['key'], extra_text_item['prob']

            txt_keys.append(key)
            c_sum += prob
            cum_sum.append(c_sum)

        txt_keys.append('txt')
        self.txt_keys = txt_keys
        self.cum_sum = np.array(cum_sum)
        self.downsample = downsample

        self.fps = fps
        self.skip_frms_num = int(skip_frms_num)
        self.image_size = image_size
        self.num_frames = num_frames
        self.bucket_helper = bucket_helper
        self.modify_prompt = modify_prompt
        self.add_stock_prefix = add_stock_prefix
        self.reader_type = reader_type
        self.use_cluster = use_cluster

        if filters is None:
            filters = []

        self.filters = filters

        super().__init__(
            path,
            self.process,
            seed=seed,
            meta_names=meta_names,
            shuffle_buffer=shuffle_buffer,
            nshards=nshards,
            include_dirs=include_dirs
        )


    def process(self, src):
        for r in src:
            if 'video_pth.zstd' in r:
                try:
                    video_buffer = r["video_pth.zstd"]
                    video_frames = deserialize_tensor_from_zstd(video_buffer).float()       #  C T H W
                    ref_frame_buffer = r["ref_frame_pth.zstd"]
                    ref_frame = deserialize_tensor_from_zstd(ref_frame_buffer).float()
                    first_frame_buffer = r["first_frame_pth.zstd"]
                    first_frame = deserialize_tensor_from_zstd(first_frame_buffer).float()
                    pixel_first_frame = r["pixel_first_frame.zstd"]
                    pixel_first_frame = deserialize_tensor_from_zstd(pixel_first_frame).float()  #  也是C T H W
                    latent_ocr_mask = r["latent_ocr_mask.zstd"]
                    latent_ocr_mask = deserialize_tensor_from_zstd(latent_ocr_mask).float()
                    if self.downsample:
                        if 'smpl_render_downsample.zstd' in r.keys():
                            smpl_render = r["smpl_render_downsample.zstd"]          # 可能需要改回去
                            smpl_render_aug = r["smpl_render_aug_downsample.zstd"]  # 可能需要改回去
                            latent_sam = r["latent_sam_mask.zstd"]
                            latent_sam = deserialize_tensor_from_zstd(latent_sam).float()
                            latent_sam_aug = r["latent_sam_aug_mask.zstd"]
                            latent_sam_aug = deserialize_tensor_from_zstd(latent_sam_aug).float()
                            latent_ref_mask = r["latent_ref_mask.zstd"]
                            latent_ref_mask = deserialize_tensor_from_zstd(latent_ref_mask).float()
                    else:
                        if 'smpl_render.zstd' in r.keys():
                            smpl_render = r["smpl_render.zstd"]
                            smpl_render_aug = r["smpl_render_aug.zstd"]

                    smpl_render = deserialize_tensor_from_zstd(smpl_render).float()
                    smpl_render_aug = deserialize_tensor_from_zstd(smpl_render_aug).float()
                    
                    
                except Exception as e:
                    print(f"error occurs when deserializing video: {e}")
                    continue
                if 'recaption' in r:
                    txt = r['recaption']
                    if isinstance(txt, bytes):
                        txt = txt.decode('utf-8')  # 假设是 UTF-8 编码
                    else:
                        txt = str(txt)
                    if txt == 'None' or txt == "":
                        continue
                else:
                    print('No recaption found')
                    continue

                # e2e_flag is used to determine augmentation prob
                e2e_flag = False
                if 'e2e_flag' in r:
                    e2e_flag = r['e2e_flag']
                    if isinstance(e2e_flag, bytes):
                        e2e_flag = e2e_flag.decode('utf-8')  # 假设是 UTF-8 编码
                    else:
                        e2e_flag = str(e2e_flag)
                    e2e_flag = (e2e_flag == 'True')

                ref_mask_flag = True
                if 'ref_mask_flag' in r:
                    ref_mask_flag = r['ref_mask_flag']
                    if isinstance(ref_mask_flag, bytes):
                        ref_mask_flag = ref_mask_flag.decode('utf-8')
                    else:
                        ref_mask_flag = str(ref_mask_flag)
                    ref_mask_flag = ('True' in ref_mask_flag)

                if 'latent_hands_mask.zstd' in r.keys():
                    latent_hands_mask_buffer = r["latent_hands_mask.zstd"]
                    latent_hands_mask = deserialize_tensor_from_zstd(latent_hands_mask_buffer).float()
                    latent_faces_mask_buffer = r["latent_faces_mask.zstd"]
                    latent_faces_mask = deserialize_tensor_from_zstd(latent_faces_mask_buffer).float()
                else:
                    print('No latent faces mask found')
                    continue
                
            else:
                print('No video data found')
                continue

            try:
                h = video_frames.shape[2]
                w = video_frames.shape[3]
                avail_frames_in_wanted_fps = video_frames.shape[1]   # C T H W
                if h is None or w is None:
                    print("no height or width")
                    continue

                if self.bucket_helper is not None:
                    bucket_id, bucket_shape = self.bucket_helper.assign_bucket(avail_frames_in_wanted_fps, h, w, is_image=False)
                    if bucket_id is None or bucket_shape is None:
                        print("no bucket available, available frames: ", avail_frames_in_wanted_fps)
                        continue
                else:
                    bucket_id = -1
                    bucket_shape = (self.num_frames,) + self.image_size

                bucket_t, bucket_h, bucket_w = bucket_shape
                

            except Exception as e:
                print(e, self.path)
                import traceback
                traceback.print_exc()
                continue

            item = {
                'mp4': video_frames, # C T H W
                'ref_frame': ref_frame, # C 1 H W
                'first_frame': first_frame, # C 1 H W
                'pixel_first_frame': pixel_first_frame, # C 1 H W
                'latent_faces_mask': latent_faces_mask,
                'latent_hands_mask': latent_hands_mask,
                'latent_ocr_mask': latent_ocr_mask,
                'txt': txt,
                'num_frames': bucket_t,
                'fps': 16,
                'bucket_id': bucket_id
            }
            if 'smpl_render.zstd' in r.keys() or 'smpl_render_downsample.zstd' in r.keys():
                item['smpl_render'] = smpl_render
                item['smpl_render_aug'] = smpl_render_aug
                item['latent_sam'] = latent_sam
                item['latent_sam_aug'] = latent_sam_aug
                item['latent_ref_mask'] = latent_ref_mask
                item['e2e_flag'] = e2e_flag
                item['ref_mask_flag'] = ref_mask_flag

            key = r.get('__key__', '')
            can_history = bool(re.search(r'_1_\d{5}', key) or re.search(r'part1_0_\d{5}', key))
            item['can_history'] = can_history
            yield item

from operator import and_
from functools import reduce

class PoseLatentBucketHelper():     # 用于测试多分辨率视频能否一起训
    def __init__(self, image_size_buckets, video_frame_batch_buckets, image_frame_batch_buckets, aspect_ratio_diff_threshold):
        self.image_size_buckets = image_size_buckets
        self.video_frame_batch_buckets = video_frame_batch_buckets
        self.image_frame_batch_buckets = image_frame_batch_buckets
        self.aspect_ratio_diff_threshold = aspect_ratio_diff_threshold

    def assign_bucket(self, num_frames, height, width, is_image):
        # choose image size bucket
        image_size_buckets = list(enumerate(self.image_size_buckets))
        if len(image_size_buckets) == 1:
            data_aspect_ratio = width / height
            bucket_aspect_ratio = image_size_buckets[0][1][1] / image_size_buckets[0][1][0]
            if max(data_aspect_ratio, bucket_aspect_ratio) / min(data_aspect_ratio, bucket_aspect_ratio) > self.aspect_ratio_diff_threshold \
                    or height < image_size_buckets[0][1][0] or width < image_size_buckets[0][1][1]:
                image_size_buckets = []
            else:
                image_size_buckets = image_size_buckets
        else:
            new_h, new_w = height, width
        
            if height > width: # portrait
                image_size_buckets = [x for x in image_size_buckets if x[1][0] >= x[1][1] and x[1][0] <= new_h]
                image_size_buckets = sorted(image_size_buckets, key=lambda x: x[1][0])

            else: # landscape
                image_size_buckets = [x for x in image_size_buckets if x[1][0] <= x[1][1] and x[1][1] <= new_w]
                image_size_buckets = sorted(image_size_buckets, key=lambda x: x[1][1])

        if not image_size_buckets: # no suitable image_size_buckets
            # print_rank0(f'no image_size_bucket suited for {num_frames}, {height}, {width}')
            return None, None
        
        image_size_bucket_id = image_size_buckets[-1][0]

        # choose frame batch bucket
        frame_batch_buckets = list(enumerate(self.image_frame_batch_buckets if is_image else self.video_frame_batch_buckets))
        frame_batch_buckets = [x for x in frame_batch_buckets if x[1][0] <= num_frames]
        frame_batch_buckets = sorted(frame_batch_buckets, key=lambda x: x[1][0])

        if not frame_batch_buckets: # no suitable frame_batch_buckets
            print_rank0(f'no frame_batch_bucket suited for {num_frames}, {height}, {width}')
            return None, None

        frame_batch_bucket_id = frame_batch_buckets[-1][0]
        final_bucket_id = image_size_bucket_id * len(self.image_frame_batch_buckets if is_image else self.video_frame_batch_buckets) + frame_batch_bucket_id

        return final_bucket_id, (self.image_frame_batch_buckets[frame_batch_bucket_id][0] if is_image else self.video_frame_batch_buckets[frame_batch_bucket_id][0], \
                                 self.image_size_buckets[image_size_bucket_id][0], \
                                 self.image_size_buckets[image_size_bucket_id][1])

class BucketHelper():
    def __init__(self, image_size_buckets, video_frame_batch_buckets, image_frame_batch_buckets, aspect_ratio_diff_threshold):
        self.image_size_buckets = image_size_buckets
        self.video_frame_batch_buckets = video_frame_batch_buckets
        self.image_frame_batch_buckets = image_frame_batch_buckets
        self.aspect_ratio_diff_threshold = aspect_ratio_diff_threshold

        self.short_edges = [min(x[0], x[1]) for x in self.image_size_buckets] # Assuming every short edges are equal
        self.short_edge = self.short_edges[0]
        assert reduce(and_, [x == self.short_edge for x in self.short_edges]), 'All short edges must be equal'

    def get_new_size(self, h, w):
        if h > w:
            new_size = (h * self.short_edge // w, self.short_edge)
        else:
            new_size = (self.short_edge, w * self.short_edge // h)
        return new_size

    def assign_bucket(self, num_frames, height, width, is_image):
        # choose image size bucket
        image_size_buckets = list(enumerate(self.image_size_buckets))
        if len(image_size_buckets) == 1:
            data_aspect_ratio = width / height
            bucket_aspect_ratio = image_size_buckets[0][1][1] / image_size_buckets[0][1][0]
            if max(data_aspect_ratio, bucket_aspect_ratio) / min(data_aspect_ratio, bucket_aspect_ratio) > self.aspect_ratio_diff_threshold \
                    or height < image_size_buckets[0][1][0] or width < image_size_buckets[0][1][1]:
                image_size_buckets = []
            else:
                image_size_buckets = image_size_buckets
        else:
            new_h, new_w = self.get_new_size(height, width)
            if new_h > height or new_w > width:
                print_rank0(f"new size {new_h}x{new_w} is larger than original size {height}x{width}")
                return None, None
        
            if height > width: # portrait
                image_size_buckets = [x for x in image_size_buckets if x[1][0] >= x[1][1] and x[1][0] <= new_h]
                image_size_buckets = sorted(image_size_buckets, key=lambda x: x[1][0])

            else: # landscape
                image_size_buckets = [x for x in image_size_buckets if x[1][0] <= x[1][1] and x[1][1] <= new_w]
                image_size_buckets = sorted(image_size_buckets, key=lambda x: x[1][1])

        if not image_size_buckets: # no suitable image_size_buckets
            # print_rank0(f'no image_size_bucket suited for {num_frames}, {height}, {width}')
            return None, None
        
        image_size_bucket_id = image_size_buckets[-1][0]

        # choose frame batch bucket
        frame_batch_buckets = list(enumerate(self.image_frame_batch_buckets if is_image else self.video_frame_batch_buckets))
        frame_batch_buckets = [x for x in frame_batch_buckets if x[1][0] <= num_frames]
        frame_batch_buckets = sorted(frame_batch_buckets, key=lambda x: x[1][0])

        if not frame_batch_buckets: # no suitable frame_batch_buckets
            print_rank0(f'no frame_batch_bucket suited for {num_frames}, {height}, {width}')
            return None, None

        frame_batch_bucket_id = frame_batch_buckets[-1][0]
        final_bucket_id = image_size_bucket_id * len(self.image_frame_batch_buckets if is_image else self.video_frame_batch_buckets) + frame_batch_bucket_id

        return final_bucket_id, (self.image_frame_batch_buckets[frame_batch_bucket_id][0] if is_image else self.video_frame_batch_buckets[frame_batch_bucket_id][0], \
                                 self.image_size_buckets[image_size_bucket_id][0], \
                                 self.image_size_buckets[image_size_bucket_id][1])


class IVBucketDataset(DataPipeline):
    def __init__(self, image_dataset_configs, video_dataset_configs, image_size_buckets, video_frame_batch_buckets, image_frame_batch_buckets,
                 video_ratio=None, image_ds_weights=None, video_ds_weights=None, aspect_ratio_diff_threshold=1.5,
                 seed=-1, batch_size=1, eval_batch_size=1,
                 bucket_helper_type='bucket_helper',
                 valid=False, **extra_kwargs):
        super().__init__()
        if seed == -1:
            seed = random.randint(0, 1000000)
        self.seed = seed
        self.video_ratio = video_ratio
        assert video_ratio is not None

        self.image_size_buckets = image_size_buckets
        self.video_frame_batch_buckets = video_frame_batch_buckets
        self.image_frame_batch_buckets = image_frame_batch_buckets
        self.video_buckets = [[] for _ in range(len(image_size_buckets) * len(video_frame_batch_buckets))]
        self.image_buckets = [[] for _ in range(len(image_size_buckets) * len(image_frame_batch_buckets))]

        if bucket_helper_type == 'pose_latent':
            helper_cls = PoseLatentBucketHelper
        elif bucket_helper_type == 'bucket_helper':
            helper_cls = BucketHelper
        else:
            raise ValueError(f'unknown bucket_helper_type: {bucket_helper_type}')
        self.bucket_helper = helper_cls(image_size_buckets=image_size_buckets, video_frame_batch_buckets=video_frame_batch_buckets, image_frame_batch_buckets=image_frame_batch_buckets, aspect_ratio_diff_threshold=aspect_ratio_diff_threshold)

        self.image_datasets = [instantiate_from_config(config, valid=valid, bucket_helper=self.bucket_helper, **{k: extra_kwargs[k] for k in extra_kwargs if k not in config.get('params', dict())}) for config in image_dataset_configs]
        self.video_datasets = [instantiate_from_config(config, valid=valid, bucket_helper=self.bucket_helper, **{k: extra_kwargs[k] for k in extra_kwargs if k not in config.get('params', dict())}) for config in video_dataset_configs]

        for d in self.image_datasets:
            assert isinstance(d, ImageWebDataset) or isinstance(d, VideoWebDataset)
        for d in self.video_datasets:
            assert isinstance(d, ImageWebDataset) or isinstance(d, VideoWebDataset) or isinstance(d, VideoPoseLatentDataset)

        if image_ds_weights is None:
            self.image_ds_weights = [1] * len(self.image_datasets)
        else:
            self.image_ds_weights = image_ds_weights
        if video_ds_weights is None:
            self.video_ds_weights = [1] * len(self.video_datasets)
        else:
            self.video_ds_weights = video_ds_weights
        assert len(self.image_ds_weights) == len(self.image_datasets)
        assert len(self.video_ds_weights) == len(self.video_datasets)

        self.image_iters = [iter(x) for x in self.image_datasets]
        self.video_iters = [iter(x) for x in self.video_datasets]

        self.batch_size = batch_size if not valid else eval_batch_size

        self.valid = valid

        if self.video_ratio < 1:
            assert len(self.image_datasets) > 0
        if self.video_ratio > 0:
            assert len(self.video_datasets) > 0

    def append(self, item, is_image): # TCHW. returns collated batch if bucket is full, otherwise returns None.
        final_bucket_id = item['bucket_id']
        frame_batch_buckets = self.image_frame_batch_buckets if is_image else self.video_frame_batch_buckets
        data_buckets = self.image_buckets if is_image else self.video_buckets

        image_size_bucket_id = final_bucket_id // len(frame_batch_buckets)
        frame_batch_bucket_id = final_bucket_id % len(frame_batch_buckets)
        
        data_buckets[final_bucket_id].append(item)

        if len(data_buckets[final_bucket_id]) >= frame_batch_buckets[frame_batch_bucket_id][1] * self.batch_size:
            return self.flush(final_bucket_id, is_image)

        return None

    def flush(self, bucket_id, is_image): # flush collated batch
        data_buckets = self.image_buckets if is_image else self.video_buckets

        new_item = default_collate(data_buckets[bucket_id])
        data_buckets[bucket_id] = []

        return new_item

    def __iter__(self):
        while True:
            if random.random() <= self.video_ratio:
                weights = self.video_ds_weights
                iters = self.video_iters
                is_image = False
            else:
                weights = self.image_ds_weights
                iters = self.image_iters
                is_image = True

            while True:
                selected_iter_tgt = random.randrange(0, sum(weights)) # Assuming weights are int list
                acc_sum = 0
                selected_iter_ptr = 0
                for i, w in enumerate(weights):
                    acc_sum += w
                    if acc_sum >= selected_iter_tgt:
                        selected_iter_ptr = i
                        break

                    selected_iter_ptr = i

                item = next(iters[selected_iter_ptr])
                new_item = self.append(item, is_image=is_image)
                if new_item is not None:
                    yield new_item
                    break

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        idx = int(path.split('_')[1])
        if 'valid' in path:
            return cls(valid=True, idx=idx, **kwargs)
        else:
            return cls(idx=idx, **kwargs)


class IVAlterDataset(DataPipeline):
    def __init__(self, video_dataset_config, video_dataset_each_filter, image_dataset_config=None,
                 video_weight=None, seed=-1, batch_size=1, eval_batch_size=1, image_bs_times=11,
                 valid=False, idx=None):
        super().__init__()
        if seed == -1:
            seed = random.randint(0, 1000000)
        self.seed = seed
        if image_dataset_config is None or video_weight == 1:
            print("Image dataset config is None, using video dataset config, set video_weight to 1.")
            video_weight = 1
            image_weight = 0
        elif video_weight == 0:
            image_weight = 1
            video_weight = 0

        if video_weight is None:
            self.video_weight = 1

        if video_weight > 0:
            video_dataset_config['params'].update(video_dataset_each_filter[idx])
            self.video_dataset = instantiate_from_config(video_dataset_config, valid=valid)
        else:
            self.video_dataset = None

        if image_weight > 0:
            self.image_dataset = instantiate_from_config(image_dataset_config)
        else:
            self.image_dataset = None

        if valid:
            self.batch_size = eval_batch_size
        else:
            self.batch_size = batch_size

        self.image_bs_times = image_bs_times
        if hasattr(self.video_dataset, 'inner_batch_size'):
            self.inner_batch_size = self.video_dataset.inner_batch_size
        else:
            self.inner_batch_size = 1
        self.video_weight = video_weight
        self.valid = valid
        self.valid_chose = "image"

    def __iter__(self):
        if self.image_dataset is not None:
            image_iter = iter(self.image_dataset)
        else:
            image_iter = None

        if self.video_dataset is not None:
            video_iter = iter(self.video_dataset)
        else:
            video_iter = None

        while True:
            sample_number = self.batch_size
            chose = "image"
            if self.valid:
                if self.image_dataset is not None and self.video_dataset is not None:
                    chose = self.valid_chose
                    self.valid_chose = "image" if self.valid_chose == "video" else "video"
                elif self.image_dataset is None:
                    chose = 'video'
                else: # self.video_dataset is None
                    chose = 'image'
            else:
                if random.random() <= self.video_weight:
                    chose = "video"

            if chose == "video":
                for _ in range(sample_number):
                    yield next(video_iter)
            else:
                buffer = []
                for _ in range(sample_number):
                    buffer.append(next(image_iter))

                buffer = default_collate(buffer)
                yield buffer
                    
    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        idx = int(path.split('_')[1])
        if 'valid' in path:
            return cls(valid=True, idx=idx, **kwargs)
        else:
            return cls(idx=idx, **kwargs)

