import argparse, os, sys, glob, yaml, math, random
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from collections import OrderedDict
from tqdm import trange, tqdm
from einops import rearrange, repeat
from functools import partial
import torch
from pytorch_lightning import seed_everything

from funcs import load_model_checkpoint, load_prompts, load_image_batch, get_filelist, save_videos
from utils.utils import instantiate_from_config

try:
    from lvdm.models.samplers.ddim import DDIMSampler
except ImportError:
    print("Warning: DDIMSampler import failed. Please ensure 'lvdm.models.samplers.ddim' exists.")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20230211, help="seed for seed_everything")
    parser.add_argument("--mode", default="base", type=str, help="which kind of inference mode: {'base', 'i2v'}")
    parser.add_argument("--ckpt_path", type=str, default=None, help="checkpoint path")
    parser.add_argument("--config", type=str, help="config (yaml) path")
    parser.add_argument("--prompt_file", type=str, default=None, help="a text file containing many prompts")
    parser.add_argument("--savedir", type=str, default=None, help="results saving path")
    parser.add_argument("--savefps", type=str, default=10, help="video fps to generate")
    parser.add_argument("--n_samples", type=int, default=1, help="num of samples per prompt", )
    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM", )
    parser.add_argument("--ddim_eta", type=float, default=1.0,
                        help="eta for ddim sampling (0.0 yields deterministic sampling)", )
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frames", type=int, default=-1, help="frames num to inference")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0,
                        help="prompt classifier-free guidance")
    parser.add_argument("--unconditional_guidance_scale_temporal", type=float, default=None,
                        help="temporal consistency guidance")
    parser.add_argument("--cond_input", type=str, default=None, help="data dir of conditional input")

    # =========================================================================
    # Latent Optimization 파라미터
    # =========================================================================
    parser.add_argument("--opt_scale", type=float, default=40.0, help="Optimization guidance scale (λ)")
    parser.add_argument("--shift_w", type=int, default=0, help="Latent shift along width (-1: pan right, +1: pan left)")
    parser.add_argument("--shift_h", type=int, default=0, help="Latent shift along height (-1: tilt up, +1: tilt down)")
    parser.add_argument("--opt_steps", type=int, default=20, help="Apply latent optimization only for the first N DDIM steps")
    parser.add_argument("--opt_iters", type=int, default=3, help="Gradient updates per optimized DDIM step")
    return parser


def calc_latent_shift_loss(latents, shift_w=0, shift_h=0):
    """
    shift_w < 0: 내용물이 왼쪽으로 이동 (Pan Right)
    shift_w > 0: 내용물이 오른쪽으로 이동 (Pan Left)
    shift_h < 0: 내용물이 아래로 이동 (Tilt Up)
    shift_h > 0: 내용물이 위로 이동 (Tilt Down)
    """
    # 이동이 전혀 없는 경우 최적화 연산 스킵
    if shift_w == 0 and shift_h == 0:
        return torch.tensor(0.0, device=latents.device, requires_grad=True)

    loss = 0.0
    frames = latents.shape[2]

    for f in range(frames - 1):
        curr_frame = latents[:, :, f, :, :]
        next_frame = latents[:, :, f + 1, :, :]

        # 1. 세로(Height) 방향 Slicing 결정
        if shift_h > 0:
            # 내용물이 위로 이동 (카메라가 아래를 봄)
            curr_h = slice(shift_h, None)      # 현재 프레임의 위쪽 자르기
            next_h = slice(None, -shift_h)     # 다음 프레임의 아래쪽 자르기
        elif shift_h < 0:
            # 내용물이 아래로 이동 (카메라가 위를 봄)
            shift = abs(shift_h)
            curr_h = slice(None, -shift)       # 현재 프레임의 아래쪽 자르기
            next_h = slice(shift, None)        # 다음 프레임의 위쪽 자르기
        else:
            curr_h = slice(None)
            next_h = slice(None)

        # 2. 가로(Width) 방향 Slicing 결정
        if shift_w > 0:
            # 내용물이 오른쪽으로 이동 (카메라가 왼쪽을 봄)
            curr_w = slice(None, -shift_w)     # 현재 프레임의 오른쪽 자르기
            next_w = slice(shift_w, None)      # 다음 프레임의 왼쪽 자르기
        elif shift_w < 0:
            # 내용물이 왼쪽으로 이동 (카메라가 오른쪽을 봄)
            shift = abs(shift_w)
            curr_w = slice(shift, None)        # 현재 프레임의 왼쪽 자르기
            next_w = slice(None, -shift)       # 다음 프레임의 오른쪽 자르기
        else:
            curr_w = slice(None)
            next_w = slice(None)

        # 3. 결정된 Slicing 범위로 텐서 자르기 및 MSE 계산
        shifted_curr = curr_frame[:, :, curr_h, curr_w]
        target_next = next_frame[:, :, next_h, next_w]

        loss += torch.nn.functional.mse_loss(target_next, shifted_curr)

    return loss / (frames - 1)


def custom_opt_ddim_sampling(model, cond, noise_shape, ddim_steps, ddim_eta, cfg_scale, opt_scale, shift_w, shift_h,
                             opt_steps=20, opt_iters=3):
    sampler = DDIMSampler(model)
    sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=ddim_eta, verbose=False)

    batch_size = noise_shape[0]
    device = model.device
    x_t = torch.randn(noise_shape, device=device)

    uc = None
    if cfg_scale != 1.0:
        uc_emb = model.get_learned_conditioning([""] * batch_size)
        if isinstance(cond, dict) and "fps" in cond:
            uc = {"c_crossattn": [uc_emb], "fps": cond["fps"]}
        else:
            uc = {"c_crossattn": [uc_emb]}

    time_range = np.flip(sampler.ddim_timesteps)
    total_steps = sampler.ddim_timesteps.shape[0]

    print(
        f"🚀 Starting Latent Optimization (Tweedie's Formula) | Steps: {ddim_steps} | Opt_Scale: {opt_scale} | Shift (H,W): ({shift_h},{shift_w})")

    for i, step in enumerate(tqdm(time_range, desc="Latent Opt Loop")):
        index = total_steps - i - 1
        ts = torch.full((batch_size,), step, device=device, dtype=torch.long)

        # -------------------------------------------------------------
        # [Phase 1] Differentiable Latent Optimization (Inner Loop 적용)
        # -------------------------------------------------------------
        x_t_opt = x_t.clone()
        num_opt_steps = opt_iters  # 스텝당 최적화 반복 횟수 (3~5 권장)

        # 디퓨전 초반(High Noise)에만 모션 가이던스 강하게 주입 (예: 앞의 20 스텝)
        if i < opt_steps:
            for opt_idx in range(num_opt_steps):
                x_t_opt = x_t_opt.detach().requires_grad_(True)

                if x_t_opt.grad is not None:
                    x_t_opt.grad.zero_()

                with torch.enable_grad():
                    # 1. UNet Forward
                    eps_pred = model.apply_model(x_t_opt, ts, cond)

                    # 2. Tweedie's Formula -> x0 예측
                    a_t = torch.tensor(sampler.ddim_alphas[index], device=device)
                    sqrt_one_minus_at = torch.tensor(sampler.ddim_sqrt_one_minus_alphas[index], device=device)
                    pred_x0 = (x_t_opt - sqrt_one_minus_at * eps_pred) / torch.sqrt(a_t)

                    # 3. Shift Loss 계산 (이전 답변의 Slicing 기반 함수 적용 필수!)
                    loss = calc_latent_shift_loss(pred_x0, shift_w=shift_w, shift_h=shift_h)
                    loss.backward()

                    grad = x_t_opt.grad.clone()

                # [핵심] Gradient Normalization: 벡터의 크기를 1로 만들고 우리가 원하는 scale을 곱함
                grad_norm = torch.norm(grad) + 1e-8
                normalized_grad = grad / grad_norm

                # 업데이트 및 분리
                x_t_opt = x_t_opt - opt_scale * normalized_grad
                x_t_opt = x_t_opt.detach()

                # (선택) 디버깅용: Loss가 얼마나 떨어지는지 확인
                # if rank == 0 and opt_idx == 0:
                #     print(f"Step {i} | Loss: {loss.item():.4f} | Grad Norm: {grad_norm.item():.4f}")
        else:
            # 20스텝 이후에는 구조가 잡혔으므로 최적화 없이 디노이징만 수행하여 화질 보존
            x_t_opt = x_t.detach()

        # -------------------------------------------------------------
        # [Phase 2] Standard DDIM Denoising (CFG 적용)
        # -------------------------------------------------------------
        with torch.no_grad():
            outs = sampler.p_sample_ddim(
                x_t_opt, cond, ts, index=index,
                use_original_steps=False,
                unconditional_guidance_scale=cfg_scale,
                unconditional_conditioning=uc
            )
            # 다음 타임스텝의 잠재 변수 업데이트
            x_t = outs[0]

    return x_t


def run_inference(args, gpu_num, gpu_no, **kwargs):
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config)
    model = model.cuda(gpu_no)
    assert os.path.exists(args.ckpt_path), f"Error: checkpoint [{args.ckpt_path}] Not Found!"
    model = load_model_checkpoint(model, args.ckpt_path)

    # VRAM 확보를 위한 가중치 프리징
    model.eval()
    model.requires_grad_(False)
    for param in model.parameters():
        param.requires_grad = False

    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    h, w = args.height // 8, args.width // 8
    frames = model.temporal_length if args.frames < 0 else args.frames
    channels = model.channels

    os.makedirs(args.savedir, exist_ok=True)

    assert os.path.exists(args.prompt_file), "Error: prompt file NOT Found!"
    prompt_list = load_prompts(args.prompt_file)
    num_samples = len(prompt_list)
    filename_list = [f"{id + 1:04d}" for id in range(num_samples)]

    samples_split = num_samples // gpu_num
    residual_tail = num_samples % gpu_num
    print(f'[rank:{gpu_no}] {samples_split}/{num_samples} samples loaded.')
    indices = list(range(samples_split * gpu_no, samples_split * (gpu_no + 1)))
    if gpu_no == 0 and residual_tail != 0:
        indices = indices + list(range(num_samples - residual_tail, num_samples))
    prompt_list_rank = [prompt_list[i] for i in indices]

    if args.mode == "i2v":
        cond_inputs = get_filelist(args.cond_input, ext='[mpj][pn][4gj]')
        assert len(
            cond_inputs) == num_samples, f"Error: conditional input ({len(cond_inputs)}) NOT match prompt ({num_samples})!"
        filename_list = [f"{os.path.split(cond_inputs[id])[-1][:-4]}" for id in range(num_samples)]
        cond_inputs_rank = [cond_inputs[i] for i in indices]

    filename_list_rank = [filename_list[i] for i in indices]

    start = time.time()
    n_rounds = len(prompt_list_rank) // args.bs
    n_rounds = n_rounds + 1 if len(prompt_list_rank) % args.bs != 0 else n_rounds

    for idx in range(0, n_rounds):
        print(f'[rank:{gpu_no}] batch-{idx + 1} ({args.bs})x{args.n_samples} ...')
        idx_s = idx * args.bs
        idx_e = min(idx_s + args.bs, len(prompt_list_rank))
        batch_size = idx_e - idx_s
        filenames = filename_list_rank[idx_s:idx_e]
        noise_shape = [batch_size, channels, frames, h, w]
        fps = torch.tensor([args.fps] * batch_size).to(model.device).long()

        prompts = prompt_list_rank[idx_s:idx_e]
        if isinstance(prompts, str):
            prompts = [prompts]

        text_emb = model.get_learned_conditioning(prompts)

        if args.mode == 'base':
            cond = {"c_crossattn": [text_emb], "fps": fps}
        elif args.mode == 'i2v':
            cond_images = load_image_batch(cond_inputs_rank[idx_s:idx_e], (args.height, args.width))
            cond_images = cond_images.to(model.device)
            img_emb = model.get_image_embeds(cond_images)
            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
            cond = {"c_crossattn": [imtext_cond], "fps": fps}
        else:
            raise NotImplementedError

        # Latent 최적화 샘플링 루프 실행
        latents = custom_opt_ddim_sampling(
            model=model,
            cond=cond,
            noise_shape=noise_shape,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
            cfg_scale=args.unconditional_guidance_scale,
            opt_scale=args.opt_scale,
            shift_w=args.shift_w,
            shift_h=args.shift_h,
            opt_steps=args.opt_steps,
            opt_iters=args.opt_iters
        )

        # 디코딩
        batch_samples = model.decode_first_stage(latents)

        # 4차원으로 풀렸을 경우를 대비한 5D (B, C, F, H, W) 복구
        if batch_samples.ndim == 4:
            batch_samples = rearrange(batch_samples, '(b f) c h w -> b c f h w', b=batch_size)

        # save_videos를 위한 6D (B, samples, C, F, H, W) 변환
        if batch_samples.ndim == 5:
            batch_samples = batch_samples.unsqueeze(1)

        # 비디오 파일 저장
        save_videos(batch_samples, args.savedir, filenames, fps=args.savefps)

    print(f"Saved in {args.savedir}. Time used: {(time.time() - start):.2f} seconds")


if __name__ == '__main__':
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    print("@CoLVDM Latent Opt Inference: %s" % now)
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)
    rank, gpu_num = 0, 1
    run_inference(args, gpu_num, rank)