import argparse
import math
import logging
import os
import random
import sys
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import imageio.v2 as imageio
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
import omegaconf
from tqdm import tqdm

from wan.configs import WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.text2video import WanT2V
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import save_video


def resolve_path(path, config_dir=None):
    if path is None:
        return None
    path = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(path):
        return path
    if config_dir is None:
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(config_dir, path))


def resolve_output_path(path, config_dir=None):
    if path is None:
        return None
    path = os.path.expandvars(os.path.expanduser(str(path)))
    if os.path.isabs(path):
        return path
    # Keep outputs relative to the launch directory by default. This makes the
    # paths shown in configs stable no matter where the config file lives.
    if path.startswith("outputs/") or path == "outputs":
        return os.path.abspath(path)
    if config_dir is None:
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(config_dir, path))


def save_config_copy(config, save_video_path):
    """Save a copy of the config file to the video output directory."""
    output_dir = os.path.dirname(save_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # Save config as YAML
    config_copy_path = os.path.join(output_dir, 'config.yaml') if output_dir else 'config.yaml'
    omegaconf.OmegaConf.save(config, config_copy_path)
    print(f"Config saved to: {config_copy_path}")

def load_video(file_path: str):
    frames = []
    reader = imageio.get_reader(file_path)
    fps = reader.get_meta_data().get("fps", 16)
    for frame in reader:
        frames.append(Image.fromarray(frame).convert("RGB"))
    reader.close()
    return frames, fps


def load_masks(mask_dir: str):
    if not os.path.isdir(mask_dir):
        raise ValueError(f"mask_dir does not exist or is not a directory: {mask_dir}")

    img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    mask_files = sorted(
        [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith(img_exts)]
    )
    if len(mask_files) == 0:
        raise ValueError(f"No mask images found in {mask_dir}")

    masks = []
    for file in mask_files:
        with Image.open(file) as img:
            masks.append(img.convert('L'))
    return masks


def preprocess_video(
    frames,
    height,
    width,
    device,
    masks=None,
):
    processed = []
    processed_masks = []
    for idx, frame in enumerate(frames):
        img = frame.resize((width, height), Image.BICUBIC)
        arr = np.array(img).astype(np.float32) / 255.0

        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        processed.append(tensor)

        if masks is not None:
            mask_img = masks[idx].resize((width, height), Image.BILINEAR)
            mask_arr = np.array(mask_img).astype(np.float32) / 255.0
            if mask_arr.ndim == 3:
                mask_arr = mask_arr[..., 0]
            processed_masks.append(torch.from_numpy(mask_arr))

    video = torch.stack(processed, dim=1)  # [C, T, H, W]

    # Apply mask after collecting all frames/masks, using a fixed gray background.
    if masks is not None:
        mask_tensor = torch.stack(processed_masks, dim=0).unsqueeze(0)  # [1, T, H, W]
        noise_bg = torch.full_like(video, 0.5)
        video = video * mask_tensor + noise_bg * (1.0 - mask_tensor)

    video = video * 2.0 - 1.0
    video = video.clamp(-1.0, 1.0)
    return video.to(device)

def preprocess_masks(masks, height, width, device):
    processed = []
    for mask in masks:
        img = mask.resize((width, height), Image.BILINEAR)
        arr = np.array(img).astype(np.float32) / 255.0
        if arr.ndim == 3:
            arr = arr[..., 0]
        tensor = torch.from_numpy(arr)
        processed.append(tensor)
    mask_tensor = torch.stack(processed, dim=0)  # [T, H, W]
    mask_tensor = mask_tensor.unsqueeze(0)  # [1, T-, H, W]
    return mask_tensor.to(device)


def resize_mask_to_latent(mask_tensor, latent):
    """Resize per-frame masks to latent shape, following VAE temporal folding (1 frame, then groups of 4)."""
    _, t_lat, h_lat, w_lat = latent.shape
    b, t_mask, h_mask, w_mask = mask_tensor.shape
    if (t_mask, h_mask, w_mask) == (t_lat, h_lat, w_lat):
        return mask_tensor

    pooled_masks = torch.zeros(
        (b, t_lat, h_lat, w_lat), device=mask_tensor.device, dtype=mask_tensor.dtype)

    def pool2d(x):
        return F.adaptive_avg_pool2d(x, (h_lat, w_lat))

    # first latent time step uses only the first frame
    pooled_masks[:, 0] = pool2d(mask_tensor[:, 0])

    # remaining latent steps each cover a chunk of 4 frames starting from frame 1
    for idx in range(1, t_lat):
        start = 1 + 4 * (idx - 1)
        end = min(start + 4, t_mask)
        if start >= t_mask:
            pooled_masks[:, idx] = pooled_masks[:, idx - 1]
            continue
        chunk = mask_tensor[:, start:end]
        pooled_masks[:, idx] = pool2d(chunk.mean(dim=1))

    return pooled_masks.clamp_(0.0, 1.0)

def get_timesteps(num_inference_steps, timesteps, strength, order, device):
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = timesteps[t_start * order :]
    return timesteps.to(device), num_inference_steps - t_start


def compute_seq_len(latent, patch_size, sp_size):
    _, t, h, w = latent.shape
    tokens = (t // patch_size[0]) * (h // patch_size[1]) * (w // patch_size[2])
    return int(np.ceil(tokens / sp_size) * sp_size)


def encode_text(encoder, prompt, device):
    return encoder([prompt], device)


def prepare_model(wan_pipe, timestep, boundary, offload_model):
    return wan_pipe._prepare_model_for_timestep(timestep, boundary, offload_model)


def run_fireflow_inversion(
    wan_pipe,
    video_latent,
    config,
    device,
    source_prompt,
    offload_model=True,
    guidance_scale=1.0,
    rank=0
):
    """Invert a clean latent X0 to structured noise X1 with midpoint integration."""
    num_inference_steps = config['inversion']['num_inverse_step']
    flow_shift = float(config.get('flow_shift', 3.0))
    strength = float(config['inversion'].get('strength', 1.0))
    
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=wan_pipe.num_train_timesteps,
        shift=flow_shift,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(num_inference_steps, device=device, shift=flow_shift)
    timesteps, _ = get_timesteps(num_inference_steps, scheduler.timesteps, strength, scheduler.order, device)
    timesteps = torch.cat([timesteps, torch.tensor([1], device=device, dtype=timesteps.dtype)])
    timesteps = timesteps.flip(0)

    boundary = wan_pipe.boundary * wan_pipe.num_train_timesteps
    seq_len = compute_seq_len(video_latent, wan_pipe.patch_size, wan_pipe.sp_size)
    neg_prompt = wan_pipe.sample_neg_prompt

    with torch.no_grad():
        if not wan_pipe.t5_cpu:
            wan_pipe.text_encoder.model.to(device)
            source_context = encode_text(wan_pipe.text_encoder, source_prompt, device)
            source_context_neg = encode_text(wan_pipe.text_encoder, neg_prompt, device)
            if offload_model:
                wan_pipe.text_encoder.model.cpu()
        else:
            cpu_device = torch.device('cpu')
            source_context = [t.to(device) for t in encode_text(wan_pipe.text_encoder, source_prompt, cpu_device)]
            source_context_neg = [t.to(device) for t in encode_text(wan_pipe.text_encoder, neg_prompt, cpu_device)]

    X = video_latent.clone()
    next_step_velocity = None
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=wan_pipe.param_dtype):
        for i in tqdm(range(len(timesteps) - 1), disable=(rank != 0), desc="Inversion steps"):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]

            t_curr_norm = t_curr.float() / wan_pipe.num_train_timesteps
            t_next_norm = t_next.float() / wan_pipe.num_train_timesteps
            dt = t_next_norm - t_curr_norm

            if next_step_velocity is None:
                timestep = torch.tensor([t_curr.item()], device=device, dtype=torch.long)
                model = prepare_model(wan_pipe, timestep, boundary, offload_model)
                pred_cond = model([X], t=timestep, context=source_context, seq_len=seq_len)[0]
                if guidance_scale != 1.0:
                    pred_uncond = model([X], t=timestep, context=source_context_neg, seq_len=seq_len)[0]
                    pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
                else:
                    pred = pred_cond
            else:
                pred = next_step_velocity

            X_mid = X + (dt / 2) * pred
            t_mid = t_curr_norm + dt / 2
            t_mid_int = int(t_mid * wan_pipe.num_train_timesteps)
            timestep_mid = torch.tensor([t_mid_int], device=device, dtype=torch.long)
            model_mid = prepare_model(wan_pipe, timestep_mid, boundary, offload_model)

            pred_mid_cond = model_mid([X_mid], t=timestep_mid, context=source_context, seq_len=seq_len)[0]
            if guidance_scale != 1.0:
                pred_mid_uncond = model_mid([X_mid], t=timestep_mid, context=source_context_neg, seq_len=seq_len)[0]
                pred_mid = pred_mid_uncond + guidance_scale * (pred_mid_cond - pred_mid_uncond)
            else:
                pred_mid = pred_mid_cond

            next_step_velocity = pred_mid
            X = X + dt * pred_mid

    if offload_model:
        wan_pipe.low_noise_model.cpu()
        wan_pipe.high_noise_model.cpu()

    return X


def run_generation_from_noise(
    wan_pipe,
    start_latent,
    config,
    device,
    prompt,
    guidance_scale,
    offload_model=True,
    rank=0,
    mask_latent=None,
    video_latent=None
):
    """Denoise latent from t=1 to t=0 with CFG using a given prompt."""
    inversion_cfg = config.get('inversion', {})
    num_inference_steps = inversion_cfg['num_denoise_step']
    flow_shift = float(config.get('flow_shift', 3.0))
    strength = float(inversion_cfg.get('strength', 1.0))

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=wan_pipe.num_train_timesteps,
        shift=flow_shift,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(num_inference_steps, device=device, shift=flow_shift)
    timesteps, _ = get_timesteps(num_inference_steps, scheduler.timesteps, strength, scheduler.order, device)
    t_init = timesteps[0].float().item() / wan_pipe.num_train_timesteps
    if rank == 0:
        print(f"Initial timestep for generation: {timesteps[0].item()} (t={t_init:.4f})")
    if mask_latent is not None:
        enable_inversion = config.get('inversion', {}).get('enable', True)
        if enable_inversion:
            # noise_std = ((1.0 - t_init)**2 + t_init**2) ** 0.5
            noise_std = 1.0
            start_latent = start_latent * mask_latent + noise_std * torch.randn_like(start_latent) * (1. - mask_latent)
        else:
            start_latent = (1.0 - t_init) * video_latent + t_init * torch.randn_like(video_latent)
            # noise_std = ((1.0 - t_init)**2 + t_init**2) ** 0.5
            noise_std = 1.0
            start_latent = start_latent * mask_latent + noise_std * torch.randn_like(start_latent) * (1. - mask_latent)
    if start_latent is None:
        start_latent = (1.0 - t_init) * video_latent + t_init * torch.randn_like(video_latent) 

    boundary = wan_pipe.boundary * wan_pipe.num_train_timesteps
    seq_len = compute_seq_len(start_latent, wan_pipe.patch_size, wan_pipe.sp_size)
    neg_prompt = wan_pipe.sample_neg_prompt

    with torch.no_grad():
        if not wan_pipe.t5_cpu:
            wan_pipe.text_encoder.model.to(device)
            context = encode_text(wan_pipe.text_encoder, prompt, device)
            context_neg = encode_text(wan_pipe.text_encoder, neg_prompt, device)
            if offload_model:
                wan_pipe.text_encoder.model.cpu()
        else:
            cpu_device = torch.device('cpu')
            context = [t.to(device) for t in encode_text(wan_pipe.text_encoder, prompt, cpu_device)]
            context_neg = [t.to(device) for t in encode_text(wan_pipe.text_encoder, neg_prompt, cpu_device)]

    X = start_latent.clone()

    with torch.no_grad(), torch.amp.autocast('cuda', dtype=wan_pipe.param_dtype):
        for i in tqdm(range(len(timesteps)), disable=(rank != 0), desc="Denoising steps"):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1] if i + 1 < len(timesteps) else torch.zeros_like(t_curr)

            t_curr_norm = t_curr.float() / wan_pipe.num_train_timesteps
            t_next_norm = t_next.float() / wan_pipe.num_train_timesteps
            dt = t_next_norm - t_curr_norm

            timestep = torch.tensor([t_curr.item()], device=device, dtype=torch.long)
            model = prepare_model(wan_pipe, timestep, boundary, offload_model)

            pred_cond = model([X], t=timestep, context=context, seq_len=seq_len)[0]
            if guidance_scale != 1.0:
                pred_uncond = model([X], t=timestep, context=context_neg, seq_len=seq_len)[0]
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                pred = pred_cond

            X = X + dt * pred

    if offload_model:
        wan_pipe.low_noise_model.cpu()
        wan_pipe.high_noise_model.cpu()

    return X


def parse_args():
    t2v_tasks = [name for name in WAN_CONFIGS.keys() if name.startswith('t2v')]
    parser = argparse.ArgumentParser(description="ProxyUp runner based on Wan2.2 native code")
    parser.add_argument('--config', type=str, required=True, help='YAML config for prompts/video paths')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Path to Wan2.2 checkpoint directory')
    parser.add_argument('--task', type=str, default='t2v-A14B', choices=t2v_tasks)
    parser.add_argument('--video', type=str, default=None, help='Override config.video.video_path')
    parser.add_argument('--mask_dir', type=str, default=None, help='Override config.video.mask_dir')
    parser.add_argument('--output', type=str, default=None, help='Override config.save_video')
    parser.add_argument('--height', type=int, default=None, help='Output video height (defaults to input video height)')
    parser.add_argument('--width', type=int, default=None, help='Output video width (defaults to input video width)')
    parser.add_argument('--offload_model', action='store_true', help='Offload DiT to CPU between steps to save VRAM')
    parser.add_argument('--t5_cpu', action='store_true', help='Place T5 encoder on CPU')
    parser.add_argument('--ulysses_size', type=int, default=1, help='Sequence parallel size (should equal world size when >1)')
    parser.add_argument('--t5_fsdp', action='store_true', help='Use FSDP for T5 encoder')
    parser.add_argument('--dit_fsdp', action='store_true', help='Use FSDP for DiT')
    parser.add_argument('--convert_model_dtype', action='store_true', help='Convert DiT parameters dtype (when not using FSDP)')
    parser.add_argument('--base_seed', type=int, default=-1, help='Base random seed; -1 for random')
    parser.add_argument('--save_intermediate_steps', type=int, default=0, help='Number of intermediate results to save (0 to disable)')
    parser.add_argument('--master_addr', type=str, default='127.0.0.1', help='Master address for distributed training')
    parser.add_argument('--master_port', type=str, default='12345', help='Port for distributed training')
    return parser.parse_args()


def main():
    warnings.filterwarnings('ignore')
    args = parse_args()
    config = omegaconf.OmegaConf.load(args.config)
    config_dir = os.path.dirname(os.path.abspath(args.config))
    video_cfg = config.get('video', {})

    video_path = args.video if args.video is not None else config['video']['video_path']
    config.video.video_path = resolve_path(video_path, None if args.video is not None else config_dir)
    if args.mask_dir is not None:
        config.video.mask_dir = resolve_path(args.mask_dir)
    if args.output is not None:
        config.save_video = resolve_output_path(args.output)

    if config.get('save_video') is None:
        raise ValueError('Config must define save_video or use --output')
    if args.output is None:
        config.save_video = resolve_output_path(config['save_video'], config_dir)

    # if args.master_port is not None:
    #     os.environ['MASTER_PORT'] = str(args.master_port)
    # if args.master_addr is not None:
    #     os.environ['MASTER_ADDR'] = str(args.master_addr)

    rank = int(os.getenv('RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    device = torch.device(f'cuda:{local_rank}') if torch.cuda.is_available() else torch.device('cpu')

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size, "ulysses_size must equal world_size when >1"
        init_distributed_group()

    if args.base_seed < 0:
        args.base_seed = random.randint(0, sys.maxsize)
    if dist.is_initialized():
        seed_list = [args.base_seed] if rank == 0 else [0]
        dist.broadcast_object_list(seed_list, src=0)
        args.base_seed = seed_list[0]
    torch.manual_seed(args.base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.base_seed)

    frames, fps = load_video(config['video']['video_path'])
    if len(frames) == 0:
        raise ValueError('No frames loaded from video path')
    if rank == 0:
        print(f"Loaded {len(frames)} frames from {config['video']['video_path']}")
        assert len(frames) % 4 == 1

    height = args.height or frames[0].height
    width = args.width or frames[0].width
    # align to multiples of 16 for VAE
    height = (height // 16) * 16
    width = (width // 16) * 16

    mask_frames = None
    mask_dir = None
    mask_tensor = None
    if isinstance(video_cfg, dict) or hasattr(video_cfg, 'get'):
        mask_dir = (
            video_cfg.get('mask_dir')
            or video_cfg.get('mask_folder')
            or video_cfg.get('mask_path')
            or video_cfg.get('mask')
        )
    if mask_dir:
        mask_dir = resolve_path(mask_dir, config_dir)
        config.video.mask_dir = mask_dir
        mask_frames = load_masks(mask_dir)
        if len(mask_frames) != len(frames):
            raise ValueError(f"Mask count ({len(mask_frames)}) does not match video frames ({len(frames)})")
        mask_tensor = preprocess_masks(mask_frames, height, width, device)
        if rank == 0:
            print(f"Loaded {len(mask_frames)} masks from {mask_dir}")

    wan_pipe = WanT2V(
        config=WAN_CONFIGS[args.task],
        checkpoint_dir=args.ckpt_dir,
        device_id=local_rank,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        init_on_cpu=True,
        convert_model_dtype=args.convert_model_dtype,
    )

    # set correct model type so DiT runs with the right conditioning path
    if hasattr(wan_pipe, 'low_noise_model'):
        setattr(wan_pipe.low_noise_model, 'model_type', 't2v')
    if hasattr(wan_pipe, 'high_noise_model'):
        setattr(wan_pipe.high_noise_model, 'model_type', 't2v')

    # Apply masks during video preprocessing before VAE encoding.
    video_tensor = preprocess_video(
        frames,
        height,
        width,
        device,
        masks=mask_frames,
    )
    video_latent = wan_pipe.vae.encode([video_tensor])[0]

    # Save config to video output directory (only on rank 0)
    if rank == 0:
        # Determine save path from the first enabled method
        save_path = config['save_video']
        if save_path:
            save_config_copy(config, save_path)

    enable_inversion = config.get('inversion', {}).get('enable', True)
    mask_latent = None
    if mask_tensor is not None and config.get('use_latent_mask', False):
        print("Resizing mask to latent shape...")
        mask_latent = resize_mask_to_latent(mask_tensor, video_latent)
        
    inversion_guidance_scale = config.get('inversion', {}).get('inverse_guidance_scale', 1.0)
    target_guidance_scale = config.get('inversion', {}).get('denoise_guidance_scale', 3.0)
    target_prompt = video_cfg.get('target_prompt', '')

    tmp_video_latent = video_latent.clone()
    if enable_inversion:
        if rank == 0:
            print('Running inversion on tmp_video_latent...')
        if mask_latent is not None:
            tmp_video_latent = video_latent * mask_latent
        inverted_noise = run_fireflow_inversion(
            wan_pipe=wan_pipe,
            video_latent=tmp_video_latent,
            config=config,
            device=device,
            source_prompt=config['inversion'].get('inverse_prompt', target_prompt),
            offload_model=args.offload_model,
            guidance_scale=inversion_guidance_scale,
            rank=rank
        )
    else:
        inverted_noise = None

    if rank == 0:
        print('Generating from inverted noise with target prompt...')
    gen_from_inv_latent = run_generation_from_noise(
        wan_pipe=wan_pipe,
        start_latent=inverted_noise,
        config=config,
        device=device,
        prompt=target_prompt,
        guidance_scale=target_guidance_scale,
        offload_model=args.offload_model,
        rank=rank,
        mask_latent=mask_latent,
        video_latent=video_latent
    )

    gen_from_inv_video = wan_pipe.vae.decode([gen_from_inv_latent])[0]
    if enable_inversion:
        inv_gen_path = config['save_video'].replace(
            '.mp4', f'_strength{config["inversion"]["strength"]}_refineSteps{config["inversion"]["refine_steps"]}_cfg{inversion_guidance_scale}-{target_guidance_scale}_seed{args.base_seed}.mp4'
        )
    else:
        inv_gen_path = config['save_video'].replace(
            '.mp4', f'_strength{config["inversion"]["strength"]}_refineSteps{config["inversion"]["refine_steps"]}_cfg{target_guidance_scale}_seed{args.base_seed}.mp4'
        )
    if mask_latent is not None:
        inv_gen_path = inv_gen_path.replace('.mp4', '_noisyBg.mp4')
    if rank == 0:
        compare_video = torch.cat([video_tensor.to(gen_from_inv_video.device), gen_from_inv_video], dim=-1)
        save_video(compare_video[None], save_file=inv_gen_path, fps=int(fps))
        print(f'Saved inversion->target generation video to {inv_gen_path}')

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
