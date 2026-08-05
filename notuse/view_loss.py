import os
import sys
import math
import argparse
import json
from typing import List, Union
from tqdm import tqdm
from omegaconf import ListConfig
from PIL import Image
from functools import partial

import torch
import numpy as np
from einops import rearrange
from torchvision.utils import make_grid
import torchvision.transforms as TT

from sgm.util import get_obj_from_str, isheatmap, exists, append_dims

from sat.model.base_model import get_model
from sat.training.model_io import load_checkpoint
from sat.data_utils import make_loaders

from diffusion_video import SATVideoDiffusionEngine
from arguments import get_args, process_config_to_args
# from data import MultiMetaWebDataset
from sgm.modules.diffusionmodules.loss import VideoDiffusionLoss, RFLoss
            

def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))

def get_batch(keys, value_dict, N: Union[List, ListConfig], device="cuda"):
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
        else:
            batch[key] = value_dict[key]
        
    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

    
def sampling_main(args, model_cls):
    if isinstance(model_cls, type):
        model = get_model(args, model_cls)
    else:
        model = model_cls

    load_checkpoint(model, args)
    def get_latest_checkpoint(checkpoint_path):
        latest_path = os.path.join(checkpoint_path, "latest")
        iteration = open(latest_path).read().strip()
        return iteration
    latest_iter = get_latest_checkpoint(args.load)
    

    data_class = get_obj_from_str(args.data_config["target"])
    create_dataset_function = partial(data_class.create_dataset_function, **args.data_config["params"])
    train_data, val_data, test_data = make_loaders(args, create_dataset_function)
    data_iter = iter(train_data)

    epochs = 50
    model.cuda()
    model.eval()
    with torch.no_grad():
        loss = {}
        for epoch_index in tqdm(range(epochs)):
            batch = next(data_iter)
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].cuda()
            if len(batch['mp4'].shape) == 6:
                b, v = batch['mp4'].shape[:2]
                batch['mp4'] = batch['mp4'].view(-1, *batch['mp4'].shape[2:])
                txt = []
                for i in range(b):
                    for j in range(v):
                        txt.append(batch['txt'][j][i])
                batch['txt'] = txt

            ## calc loss of certain sigma
            x = model.get_input(batch)
            x = model.encode_first_stage(x, batch)
            x = x.permute(0, 2, 1, 3, 4).contiguous() # b t c h w

            cond = model.conditioner(batch)
            additional_model_inputs = {
                key: batch[key] for key in model.loss_fn.batch2model_keys.intersection(batch)
            }
            print(batch['txt'])

            eval_step_list = list(range(100, 1000, 200))
            for step in eval_step_list:
                if epoch_index == 0:
                    loss[step] = 0
                
                noise = torch.randn_like(x)
                if isinstance(model.loss_fn, VideoDiffusionLoss):
                    raise NotImplementedError()
                    alphas_cumprod_sqrt_schedule = torch.load('/path/to/sat_sdxl/shift-1.0.pt')
                    alphas_cumprod_sqrt = alphas_cumprod_sqrt_schedule[[step]*x.shape[0]].to(x.device)
                    additional_model_inputs['idx'] = (alphas_cumprod_sqrt - model.loss_fn.sigma_sampler.sigmas.to(x.device)[:,None]).abs().argmin(dim=0)

                    noised_input = x.float() * append_dims(alphas_cumprod_sqrt, x.ndim) + noise * append_dims((1-alphas_cumprod_sqrt**2)**0.5, x.ndim)
                    model_output = model.denoiser(
                        model.model, noised_input, alphas_cumprod_sqrt, cond, **additional_model_inputs
                    )
                    w = append_dims(1/(1-alphas_cumprod_sqrt**2), x.ndim) # v-pred
                    loss[step] += model.loss_fn.get_loss(model_output, x, w).to(torch.float64).mean().cpu().item()

                elif isinstance(model.loss_fn, RFLoss):
                    all_sigmas = torch.tensor(np.linspace(1, 0, 1000), dtype=torch.float32, device=x.device)
                    sigma = all_sigmas[[step]*x.shape[0]]
                    noised_input = x.float() * append_dims(1 - sigma, x.ndim) + noise * append_dims(sigma, x.ndim)
                    model_output = model.denoiser(
                        model.model, noised_input, sigma, cond, **additional_model_inputs
                    )
                    w = 1
                    loss[step] += model.loss_fn.get_loss(model_output, noise - x, w).to(torch.float64).mean().cpu().item()
        
        for step in eval_step_list:
            loss[step] /= epochs
        print(loss)
        save_dir = f'view_loss/ablation/{args.load}'
        os.makedirs(save_dir, exist_ok=True)
        with open(f'{save_dir}/{latest_iter}', 'w') as f:
            f.write('loss\n')
            f.write(json.dumps(loss)+'\n')



if __name__ == '__main__':
    py_parser = argparse.ArgumentParser(add_help=False)
    known, args_list = py_parser.parse_known_args()

    args = get_args(args_list)
    args = argparse.Namespace(**vars(args), **vars(known))
    args.model_config.first_stage_config.params.cp_size = 1
    args.model_config.network_config.params.transformer_args.model_parallel_size = 1
    args.model_config.network_config.params.transformer_args.checkpoint_activations = False
    sampling_main(args, model_cls=SATVideoDiffusionEngine)
