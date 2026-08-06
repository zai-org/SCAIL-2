# SCAIL-2 (SAT Implementation)

This branch holds the original **SAT-based** implementation of SCAIL-2 used to produce the results reported in the paper. It is preserved for reproducibility. For day-to-day inference, use the streamlined [`wan-scail2`](https://github.com/zai-org/SCAIL-2/tree/wan-scail2) branch instead.

## Checkpoints

| ckpts       | Download Link                                                                                                                |    Notes                      |
|--------------|------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| SCAIL-2 | [🤗 Hugging Face](https://huggingface.co/zai-org/SCAIL-2) <br> [🤖 ModelScope](https://modelscope.cn/models/ZhipuAI/SCAIL-2) | Trained with mixed resolutions and fps. <br> End-to-end driven supports both 512p and 704p. <br> Pose-driven performs better under 704p.  <br> H and W should be both divisible by 32<br> (e.g. 704*1280) if using other resolutions. |

The checkpoint integrates Wan VAE and T5; arrange the downloaded files as:

```
SCAIL-2/
├── Wan2.1_VAE.pth
├── model
│   ├── 1
│   │   └── fsdp2_rank_0000_checkpoint.pt
│   └── latest
└── umt5-xxl
    ├── ...
```

## Environment

Python 3.10–3.12.

```
pip install -r requirements.txt
```

## Driving Video & Mask Preparation

SCAIL-2 takes three driving signals in addition to the reference image: a *driving video*, a per-frame *driving mask*, and a *reference mask*. Use the `scail_pose` submodule to generate them:

```shell
git submodule update --init --recursive
cd scail_pose
# follow instructions in POSE_INSTRUCTION.md
```

Depending on the driving mode, the files produced differ:

- **End-to-end driven (recommended).** `rendered_v2.mp4` is a copy of `driving.mp4` — the model consumes the raw driving frames directly. The pipeline still produces `rendered_mask_v2.mp4` (per-frame foreground mask of the driver) and `ref_mask.jpg` (foreground mask of the reference image).
- **Pose-driven.** `rendered_v2.mp4` is an SMPL pose-rendered video derived from the driving video, paired with `rendered_mask_v2.mp4` / `ref_mask.jpg` as above.
- **Cross-identity replacement.** Instead of `rendered_mask_v2.mp4`, supply `replace_mask.mp4` (the region to be replaced) together with `ref_mask.jpg`.

Each example directory should look like:

```
examples/
├── 001
│   ├── driving.mp4
│   ├── ref.jpg
│   ├── rendered_v2.mp4         # end-to-end: copy of driving.mp4; pose-driven: SMPL rendering
│   ├── rendered_mask_v2.mp4    # OR replace_mask.mp4 (cross-identity replacement)
│   └── ref_mask.jpg            # foreground mask of the reference image
└── 002
...
```

## Inference

CLI input:

```
bash scripts/sample_sgl_14Bsc_xc_cli.sh
```

The CLI accepts entries in the form `<prompt>@@<example_dir>`, e.g. `the girl is dancing@@examples/001`. Results are written to `samples/`.

Txt input: set `input_file` in [`configs/sampling/wan_pose_14Bsc_xc_txt.yaml`](configs/sampling/wan_pose_14Bsc_xc_txt.yaml) to a file with the same `<prompt>@@<example_dir>` format, then run:

```
bash scripts/sample_sgl_14Bsc_xc_txt.sh
```

The model is trained with **long detailed prompts**; short or empty prompts will run but produce weaker results. Sampling configurations (resolution, etc.) live in `configs/sampling/`; for custom sampling logic edit `sample_video.py`.


## Training
This repository supports SCAIL-2 training with DeepSpeed ZeRO-2 and FSDP2 on cached latent WebDataset shards. Build the training cache with the [`wan-scail2`](https://github.com/zai-org/SCAIL-2/tree/wan-scail2) branch first, using its `cache_scail2_wds.py` workflow; that cache script runs the target video, driving signal, reference image, and masks through the Wan VAE and writes the fields consumed by `data_video.VideoPoseLatentDataset`.

```sh
bash scripts/train_mpi_14Bsc_xc_latent_example.sh

bash scripts/train_mpi_14Bsc_xc_latent_fsdp_example.sh
```

Set `SCAIL2_POSE_LATENT_TRAIN_DIR` to the cached WDS directory and `SCAIL2_INIT_CKPT` to an initial SAT checkpoint if you are fine-tuning from a local checkpoint. The example configs keep `wandb` disabled by default and use placeholder data paths.

## Acknowledgements

Built on [Wan 2.1](https://github.com/Wan-Video/Wan2.1); project architecture inherited from [SCAIL](https://github.com/zai-org/SCAIL).

## Citation

```bibtex
@article{yan2025scail,
  title={SCAIL: Towards Studio-Grade Character Animation via In-Context Learning of 3D-Consistent Pose Representations},
  author={Yan, Wenhao and Ye, Sheng and Yang, Zhuoyi and Teng, Jiayan and Dong, ZhenHui and Wen, Kairui and Gu, Xiaotao and Liu, Yong-Jin and Tang, Jie},
  journal={arXiv preprint arXiv:2512.05905},
  year={2025}
}
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
