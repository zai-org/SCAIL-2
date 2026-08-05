import os
import argparse
from functools import partial
from PIL import Image
import numpy as np
import torch.distributed
import torchvision
import imageio

import torch

from sat import mpu
from sat.training.deepspeed_training import training_main

from sgm.util import get_obj_from_str, isheatmap

import diffusion_video
from arguments import get_args

from glob import glob
from einops import rearrange
try:
    import wandb
except ImportError:
    print("warning: wandb not installed")

def debatch_collate(items):
    return items[0]

def print_debug(args, s):
    if args.debug:
        s = f"RANK:[{torch.distributed.get_rank()}]:" + s
        print(s)

def save_texts(texts, save_dir, iterations):
    output_path = os.path.join(save_dir, f"{str(iterations).zfill(8)}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')

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
        if args is not None and args.wandb:
            wandb.log({key+f'_video_{i}': wandb.Video(now_save_path, fps=fps, format="mp4")}, step=args.iteration+1)


def log_video(batch, model, args, only_log_video_latents=False):
    # if not os.path.exists(os.path.join(args.save, 'training_config.yaml')):
    #     configs = [OmegaConf.load(cfg) for cfg in args.base]
    #     config = OmegaConf.merge(*configs)
    #     os.makedirs(args.save, exist_ok=True)
    #     OmegaConf.save(config=config, f=os.path.join(args.save, 'training_config.yaml'))

    texts = batch['txt']
    text_save_dir = os.path.join(args.save, "video_texts")
    os.makedirs(text_save_dir, exist_ok=True)
    save_texts(texts, text_save_dir, args.iteration)

    gpu_autocast_kwargs = {
        "enabled": torch.is_autocast_enabled(),  # torch.is_autocast_enabled(),
        "dtype": torch.get_autocast_gpu_dtype(),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }
    with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
        videos = model.log_video(batch, only_log_video_latents=only_log_video_latents)

    if torch.distributed.get_rank() == 0:
        root = os.path.join(args.save, "video")

        if batch['fps'].dim() == 0:
            fps = batch['fps'].cpu().item()
        else:
            fps = batch['fps'][0].cpu().item()
            
        if only_log_video_latents:
            root = os.path.join(root, "latents")
            filename = "{}_gs-{:06}".format(
                    'latents', args.iteration
                )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            os.makedirs(path, exist_ok=True)
            torch.save(videos['latents'], os.path.join(path, 'latent.pt'))
        else:
            for k in videos:
                N = videos[k].shape[0]
                if not isheatmap(videos[k]):
                    videos[k] = videos[k][:N]
                if isinstance(videos[k], torch.Tensor):
                    videos[k] = videos[k].detach().float().cpu()
                    if not isheatmap(videos[k]):
                        videos[k] = torch.clamp(videos[k], -1.0, 1.0)
            for k in videos:
                samples = (videos[k] + 1.0) / 2.0
                filename = "{}_gs-{:06}".format(
                    k, args.iteration
                )

            if batch['fps'].dim() == 0:
                fps = batch['fps'].cpu().item()
            else:
                fps = batch['fps'][0].cpu().item()
        

            if only_log_video_latents:
                root = os.path.join(root, "latents")
                filename = "{}_gs-{:06}".format(
                        'latents', args.iteration
                    )
                path = os.path.join(root, filename)
                os.makedirs(os.path.split(path)[0], exist_ok=True)
                os.makedirs(path, exist_ok=True)
                torch.save(videos['latents'], os.path.join(path, 'latents.pt'))
            else:
                for k in videos:
                    samples = (videos[k] + 1.0) / 2.0
                    filename = "{}_gs-{:06}".format(
                        k, args.iteration
                    )

                    path = os.path.join(root, filename)
                    os.makedirs(os.path.split(path)[0], exist_ok=True)
                    save_video_as_grid_and_mp4(samples, path, fps, args, k)

def log_image(batch, model, args):

    gpu_autocast_kwargs = {
        "enabled": torch.is_autocast_enabled(),  # torch.is_autocast_enabled(),
        "dtype": torch.get_autocast_gpu_dtype(),
        "cache_enabled": torch.is_autocast_cache_enabled(),
    }

    with torch.no_grad(), torch.cuda.amp.autocast(**gpu_autocast_kwargs):
        images = model.log_video(batch)

    for k in images:
        if isinstance(images[k], torch.Tensor):
            images[k] = images[k].detach().float().cpu()
            images[k] = torch.clamp(images[k], -1.0, 1.0)

    if torch.distributed.get_rank() == 0:
        texts = batch['txt']
        text_save_dir = os.path.join(args.save, "texts")
        os.makedirs(text_save_dir, exist_ok=True)
        save_texts(texts, text_save_dir, args.iteration)

        root = os.path.join(args.save, "images")
        for k in images:
            if len(images[k].shape) == 5:
                images[k] = images[k].squeeze(1)
            grid = torchvision.utils.make_grid(images[k], nrow=2)
            grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}_gs-{:06}.png".format(
                k, args.iteration
            )
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            img = Image.fromarray(grid)
            img.save(path)
            if args is not None and args.wandb:
                wandb.log({k: wandb.Image(path)}, step=args.iteration+1)

def broadcast_batch(batch):
    group = mpu.get_data_broadcast_group()
    src = mpu.get_data_broadcast_src_rank()
    rank = mpu.get_data_broadcast_rank()

    if rank == 0:
        num_keys = [len(batch.keys())]
    else:
        num_keys = [None]
    torch.distributed.broadcast_object_list(num_keys, src=src, group=group)

    if rank == 0:
        key_shape_obj_list = [(key, batch[key].shape if isinstance(batch[key], torch.Tensor) else batch[key]) for key in batch.keys()] # torch.Shape for Tensor, obj for others
    else:
        key_shape_obj_list = [None] * num_keys[0]
    torch.distributed.broadcast_object_list(key_shape_obj_list, src=src, group=group)

    for key_shape_obj in key_shape_obj_list:
        key, shape_or_obj = key_shape_obj
        if isinstance(shape_or_obj, torch.Size): # torch.Tensor
            if rank != 0:
                batch[key] = torch.zeros(shape_or_obj, device='cuda')
            torch.distributed.broadcast(batch[key], src=src, group=group)

        else: # object
            if rank != 0:
                batch[key] = shape_or_obj

    return batch

def forward_step_eval(data_iterator, model, args, timers, data_class=None, only_log_video_latents=False):
    if mpu.get_model_parallel_rank() == 0 and mpu.get_sequence_parallel_rank() == 0:
        timers('data loader').start()
        batch_image = next(data_iterator)
        batch_video = next(data_iterator)
        timers('data loader').stop()
        if batch_video["fps"].sum() == 0:
            tmp = batch_video
            batch_video = batch_image
            batch_image = tmp

        for key in batch_image:
            if isinstance(batch_image[key], torch.Tensor):
                batch_image[key] = batch_image[key].cuda()
        for key in batch_video:
            if isinstance(batch_video[key], torch.Tensor):
                batch_video[key] = batch_video[key].cuda()
    else:
        batch_video = {}
        batch_image = {}

    broadcast_batch(batch_video)
    broadcast_batch(batch_image)
    if mpu.get_data_parallel_rank() == 0:
        # try:
        log_video(batch_video, model, args, only_log_video_latents=only_log_video_latents)
        # if not only_log_video_latents:
        #     log_image(batch_image, model, args)
        # except Exception as e:
        #     print(e)
        #     pass

    batch_video['global_step'] = args.iteration
    loss, loss_dict = model.shared_step(batch_video)
    for k in loss_dict:
        if loss_dict[k].dtype == torch.bfloat16:
            loss_dict[k] = loss_dict[k].to(torch.float32)
    return loss, loss_dict

def forward_step(data_iterator, model, args, timers, data_class=None):

    if mpu.get_model_parallel_rank() == 0 and mpu.get_sequence_parallel_rank() == 0:
        timers('data loader').start()
        batch = next(data_iterator)
        timers('data loader').stop()
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].cuda()
    else:
        batch = {}

    batch['global_step'] = args.iteration
    broadcast_batch(batch)

    log_dict = {}
    log_dict['real_samples'] = batch['mp4'].shape[0] # b t c h w
    log_dict['real_video_samples'] = batch['mp4'].shape[0] if batch['mp4'].shape[1] > model.model.diffusion_model.patch_size[0] else 0
    for key in log_dict.keys():
        log_dict[key] = torch.tensor(log_dict[key]).cuda()
        torch.distributed.all_reduce(log_dict[key], group=mpu.get_data_parallel_group())
        log_dict[key] = log_dict[key].cpu().item()

    loss, loss_dict = model.shared_step(batch)

    return loss, loss_dict, log_dict

root_path = 'batch_save'

def save_batch(batch):
    import datetime
    path = os.path.join(root_path, datetime.datetime.now().__str__() + '_rank%d' % torch.distributed.get_rank())
    b, t = batch['mp4'].shape[:2]
    for bb in range(b):
        b_path = os.path.join(path, str(bb))
        os.makedirs(b_path, exist_ok=True)
        txt = batch['txt'][bb]
        f = open(os.path.join(b_path, 'txt.txt'), 'w')
        f.write(txt)
        f.close()

        for tt in range(t):
            frame = ((rearrange(batch['mp4'][bb, tt, ::], 'c h w -> h w c') + 1) * 128).clamp(0, 255).detach().cpu().numpy().astype(dtype='uint8')
            img = Image.fromarray(frame, 'RGB')
            img.save(os.path.join(b_path, '%d.jpg' % tt))


if __name__ == '__main__':
    if 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
        os.environ['LOCAL_RANK'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
        os.environ['WORLD_SIZE'] = os.environ['OMPI_COMM_WORLD_SIZE']
        os.environ['RANK'] = os.environ['OMPI_COMM_WORLD_RANK']
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()
    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))

    data_class = get_obj_from_str(args.data_config["target"])
    create_dataset_function = partial(data_class.create_dataset_function, **args.data_config["params"])

    import yaml
    configs = []
    for config in args.base:
        with open(config, 'r') as f:
            base_config = yaml.safe_load(f)
        configs.append(base_config)
    args.log_config = configs

    if args.model_type == "dit":
        Engine = diffusion_video.SATVideoDiffusionEngine
    else:
        raise NotImplementedError(f"model_type={args.model_type!r} is not supported in this training release")
    
    training_main(args, model_cls=Engine,
        forward_step_function=partial(forward_step, data_class=data_class), 

        forward_step_eval=partial(forward_step_eval, data_class=data_class, only_log_video_latents=args.only_log_video_latents),
        create_dataset_function=create_dataset_function, collate_fn=debatch_collate)
