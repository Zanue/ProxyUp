# ProxyUp: Training-Free Proxy-Conditioned Video Generation for Controllable Dynamics

### [Project Page](https://zanue.github.io/proxyup/) | Arxiv Paper

[Zanwei Zhou](https://scholar.google.com/citations?user=45oVlf8AAAAJ&hl=en)<sup>1,* </sup>, [Jiazhong Cen](https://jumpat.github.io/jumpcat/)<sup>1,* </sup>, [Jiemin Fang](https://jaminfong.cn/)<sup>2,&dagger;</sup>, [Yumeng He](https://github.com/raynehe)<sup>1</sup>, [Chen Yang](https://chensjtu.github.io/)<sup>2</sup>, Sikuang Li<sup>1</sup>, Fanpeng Meng<sup>2</sup>, Zhikuan Bao<sup>2</sup>, [Wei Shen](https://shenwei1231.github.io/)<sup>1,&dagger;</sup>, [Qi Tian](https://www.qitian1987.com/)<sup>2,&dagger;</sup>

<sup>1</sup>Shanghai Jiao Tong University &nbsp;&nbsp; <sup>2</sup>Huawei Inc.

<sup>*</sup>Equal contribution. Work done during internship at Huawei. &nbsp;&nbsp; <sup>&dagger;</sup>Corresponding author.

ProxyUp is a training-free framework for proxy-conditioned controllable video generation. Given a coarse proxy video from physics simulation, graphics rendering, or real-world recording, together with a text prompt, ProxyUp synthesizes a new video that preserves the proxy dynamics while regenerating prompt-aligned visual content.

## Updates

- 2026/07/03: Release initial inference code, demo configurations, and example proxy videos.


## Setup

This implementation is built on top of Wan2.2. Please install the official Wan2.2 codebase and dependencies first:

- Wan2.2 repository: [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)
- Wan2.2 T2V-A14B checkpoint: [Wan-AI/Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B)

After installation, make sure the `wan` Python package is importable in your environment. Then place or download the Wan2.2 checkpoint that will be passed to `--ckpt_dir`.

The demo commands below assume that they are launched from this `code/` directory.

## Inference

Run a demo configuration:

```bash
python run_proxyup.py \
  --config configs/demos/newtons_cradle.yaml \
  --ckpt_dir /path/to/Wan2.2-T2V-A14B \
  --task t2v-A14B \
  --offload_model \
  --t5_cpu
```

The script saves a side-by-side comparison video to the `save_video` path specified in the config. It also writes a copy of the resolved configuration next to the output.

You may reuse a configuration with a different proxy video or output path:

```bash
python run_proxyup.py \
  --config configs/demos/box_open.yaml \
  --video /path/to/your_proxy.mp4 \
  --output outputs/my_proxyup_result.mp4 \
  --ckpt_dir /path/to/Wan2.2-T2V-A14B
```

For multi-GPU sequence-parallel inference, launch with `torchrun` and set `--ulysses_size` to the world size:

```bash
torchrun --nproc_per_node=4 run_proxyup.py \
  --config configs/demos/curtain_open.yaml \
  --ckpt_dir /path/to/Wan2.2-T2V-A14B \
  --ulysses_size 4
```

## Demo Configs

| Config | Proxy video | Motion prompt |
| --- | --- | --- |
| `configs/demos/newtons_cradle.yaml` | `demos/newtons_cradle_proxy_video.mp4` | Newton's Cradle |
| `configs/demos/curtain_open.yaml` | `demos/curtain_open_proxy_video.mp4` | curtain opening |
| `configs/demos/cut_bread.yaml` | `demos/cut_bread_proxy_video.mp4` | bread slicing |
| `configs/demos/milk_stir.yaml` | `demos/milk_proxy_video.mp4` | milk stirring |
| `configs/demos/book_fall_down.yaml` | `demos/book_fall_down_proxy_video.mp4` | falling books |
| `configs/demos/box_open.yaml` | `demos/box_proxy_video.mp4` | box opening |
| `configs/demos/fridge_open.yaml` | `demos/fridge_proxy_video.mp4` | refrigerator opening |
| `configs/demos/stove_open.yaml` | `demos/stove_proxy_video.mp4` | oven opening |
| `configs/demos/table_drawer.yaml` | `demos/table_proxy_video.mp4` | drawer opening |

## Config Notes

Important fields:

- `video.video_path`: path to the proxy video. Relative paths are resolved from the config file location.
- `video.target_prompt`: final generation prompt.
- `video.mask_dir`: frame-aligned grayscale masks. Replace `/path/to/your/mask` with your mask folder.
- `save_video`: output path. Paths under `outputs/` are resolved from the launch directory.
- `inversion`: inversion, region-wise latent noising, and denoising controls.

Input videos should have `4n + 1` frames, matching Wan's temporal VAE layout.

## Repository Layout

```text
.
├── run_proxyup.py
├── configs/
│   └── demos/
└── demos/
    ├── *_proxy_video.mp4
```

## Acknowledgement

This project is built upon [Wan2.2](https://github.com/Wan-Video/Wan2.2). We sincerely appreciate the excellent work of the Wan team and the open-source community.

## Citation

If you find this repository helpful in your research, please consider citing our paper and giving this repository a star. 
