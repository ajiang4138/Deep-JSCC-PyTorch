#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from constellation import map_to_mic_codebook, pair_channels_to_symbols
from mic_mapper_visualizer import (build_model_from_checkpoint,
                                   infer_default_shape, load_input_tensor,
                                   plot_single_constellation,
                                   tensor_image_to_numpy)


def compute_psnr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    mse = float(np.mean((x - y) ** 2))
    if mse <= eps:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def kmeans_codebook(
    points: np.ndarray,
    k: int,
    seed: int,
    max_iters: int = 30,
    max_points_for_fit: int = 30000,
) -> np.ndarray:
    if points.shape[0] == 0:
        raise ValueError('Empty point set for k-means codebook estimation')

    rng = np.random.default_rng(seed)

    fit_points = points
    if fit_points.shape[0] > max_points_for_fit:
        idx = rng.choice(fit_points.shape[0], size=max_points_for_fit, replace=False)
        fit_points = fit_points[idx]

    k = int(max(2, min(k, fit_points.shape[0])))
    init_idx = rng.choice(fit_points.shape[0], size=k, replace=False)
    centers = fit_points[init_idx].copy()
    labels = np.zeros(fit_points.shape[0], dtype=np.int64)

    for _ in range(max_iters):
        d2 = ((fit_points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        new_centers = centers.copy()
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                new_centers[i] = fit_points[mask].mean(axis=0)
            else:
                new_centers[i] = fit_points[rng.integers(0, fit_points.shape[0])]
        centers = new_centers

    return centers


def estimate_codebook_from_latent(
    z_pre: torch.Tensor,
    k: int,
    seed: int,
    clip_value: float,
) -> np.ndarray:
    sym = pair_channels_to_symbols(z_pre.detach().cpu()).reshape(-1, 2).numpy()
    sym = np.clip(sym, -clip_value, clip_value)

    rounded = np.round(sym, decimals=5)
    unique = np.unique(rounded, axis=0)
    if 2 <= unique.shape[0] <= k:
        return unique

    return kmeans_codebook(sym, k=k, seed=seed)


def latent_to_iq_points(z_tensor: torch.Tensor, max_points: int = 50000) -> np.ndarray:
    points = pair_channels_to_symbols(z_tensor.detach().cpu()).reshape(-1, 2).numpy()
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
    return points


def save_triplet_plot(
    input_img: np.ndarray,
    no_mic_img: np.ndarray,
    mic_img: np.ndarray,
    psnr_no_mic: float,
    psnr_mic: float,
    out_path: Path,
):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(input_img)
    axes[0].set_title('Input')
    axes[0].axis('off')

    axes[1].imshow(no_mic_img)
    axes[1].set_title('Output w/o MIC\nPSNR {:.2f} dB'.format(psnr_no_mic))
    axes[1].axis('off')

    axes[2].imshow(mic_img)
    axes[2].set_title('Output w/ MIC\nPSNR {:.2f} dB'.format(psnr_mic))
    axes[2].axis('off')

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches='tight')
    plt.close(fig)


def run_for_image(
    model,
    image_path: str,
    out_dir: Path,
    fallback_hw: Tuple[int, int],
    args,
    image_index: int,
):
    resize_hw = (args.resize_h, args.resize_w) if args.resize_h > 0 and args.resize_w > 0 else None
    x, input_source = load_input_tensor(image_path, args.device_obj, resize_hw, fallback_hw)

    model.eval()
    with torch.no_grad():
        z_pre = model.encoder(x)

        z_no_mic = z_pre
        if hasattr(model, 'channel') and model.channel is not None:
            torch.manual_seed(args.seed + image_index)
            z_no_mic = model.channel(z_no_mic)
        x_hat_no_mic = model.decoder(z_no_mic)

        used_trained_mapper = bool(model.mapper is not None and args.prefer_trained_mapper)
        if used_trained_mapper:
            z_mic = model.mapper(z_pre)
            codebook = model.mapper.get_effective_codebook().detach().cpu().numpy() if hasattr(model.mapper, 'get_effective_codebook') else None
            mic_mode = 'trained_mapper'
        else:
            codebook = estimate_codebook_from_latent(
                z_pre=z_pre,
                k=args.mic_constellation_size,
                seed=args.seed + image_index,
                clip_value=args.mic_clip_value,
            )
            codebook_t = torch.tensor(codebook, dtype=z_pre.dtype, device=z_pre.device)
            z_mic, _ = map_to_mic_codebook(
                z_tensor=z_pre,
                codebook=codebook_t,
                clip_value=args.mic_clip_value,
                power_constraint_mode=args.mic_power_constraint_mode,
            )
            mic_mode = 'estimated_codebook_mapper'

        if hasattr(model, 'channel') and model.channel is not None:
            torch.manual_seed(args.seed + image_index)
            z_mic = model.channel(z_mic)
        x_hat_mic = model.decoder(z_mic)

    x_np = tensor_image_to_numpy(x)
    no_np = tensor_image_to_numpy(x_hat_no_mic)
    mic_np = tensor_image_to_numpy(x_hat_mic)
    pre_pts = latent_to_iq_points(z_pre)
    post_pts = latent_to_iq_points(z_mic)

    psnr_no = compute_psnr(x_np, no_np)
    psnr_mic = compute_psnr(x_np, mic_np)

    image_stem = Path(image_path).stem
    sample_dir = out_dir / image_stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(sample_dir / 'input_image.npy'), x.detach().cpu().numpy())
    np.save(str(sample_dir / 'x_hat_no_mic.npy'), x_hat_no_mic.detach().cpu().numpy())
    np.save(str(sample_dir / 'x_hat_mic.npy'), x_hat_mic.detach().cpu().numpy())
    np.save(str(sample_dir / 'mic_z_pre_mapper_iq_points.npy'), pre_pts)
    np.save(str(sample_dir / 'mic_z_post_mapper_iq_points.npy'), post_pts)

    if codebook is not None:
        np.save(str(sample_dir / 'mic_codebook_used.npy'), codebook)

    plot_single_constellation(
        pre_pts=pre_pts,
        post_pts=post_pts,
        codebook=codebook,
        centroid_info=None,
        out_path=sample_dir / 'mic_constellation.png',
    )

    save_triplet_plot(
        input_img=x_np,
        no_mic_img=no_np,
        mic_img=mic_np,
        psnr_no_mic=psnr_no,
        psnr_mic=psnr_mic,
        out_path=sample_dir / 'triplet_input_no_mic_with_mic.png',
    )

    metadata = {
        'input_source': input_source,
        'image_name': image_path,
        'mic_mode': mic_mode,
        'checkpoint': args.checkpoint,
        'used_trained_mapper': used_trained_mapper,
        'has_mapper_in_model': bool(model.mapper is not None),
        'mic_constellation_size': int(codebook.shape[0]) if codebook is not None else None,
        'mic_clip_value': args.mic_clip_value,
        'mic_power_constraint_mode': args.mic_power_constraint_mode,
        'psnr_no_mic_db': psnr_no,
        'psnr_mic_db': psnr_mic,
        'saved_files': [
            'triplet_input_no_mic_with_mic.png',
            'mic_constellation.png',
            'input_image.npy',
            'x_hat_no_mic.npy',
            'x_hat_mic.npy',
            'mic_z_pre_mapper_iq_points.npy',
            'mic_z_post_mapper_iq_points.npy',
        ],
    }
    if codebook is not None:
        metadata['saved_files'].append('mic_codebook_used.npy')

    with open(sample_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return {
        'image': image_path,
        'sample_dir': str(sample_dir),
        'psnr_no_mic_db': psnr_no,
        'psnr_mic_db': psnr_mic,
        'mic_mode': mic_mode,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Generate input/no-MIC/with-MIC image triplets')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint used to reconstruct images')
    parser.add_argument('--images', nargs='+', type=str, required=True,
                        help='one or more input image paths')
    parser.add_argument('--output_dir', type=str, default='./out/test_mic_triplet',
                        help='directory to save triplet outputs')

    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--inner_channel', type=int, default=None)
    parser.add_argument('--channel', type=str, default=None, choices=['AWGN', 'Rayleigh'])
    parser.add_argument('--snr', type=float, default=None)

    parser.add_argument('--resize_h', type=int, default=0)
    parser.add_argument('--resize_w', type=int, default=0)

    parser.add_argument('--prefer_trained_mapper', action='store_true',
                        help='if checkpoint contains mapper, use it for MIC path; otherwise estimate codebook')
    parser.add_argument('--mic_constellation_size', type=int, default=16,
                        help='constellation size used when estimating MIC codebook')
    parser.add_argument('--mic_clip_value', type=float, default=2.0)
    parser.add_argument('--mic_power_constraint_mode', type=str, default='none',
                        choices=['none', 'codebook', 'post_mapper'])

    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, switching to CPU.')
        args.device = 'cpu'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.device_obj = torch.device(args.device)

    model, meta, _ = build_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device_obj,
        inner_channel_override=args.inner_channel,
        channel_override=args.channel,
        snr_override=args.snr,
    )

    fallback_hw = infer_default_shape(meta)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, image_path in enumerate(args.images):
        result = run_for_image(
            model=model,
            image_path=image_path,
            out_dir=out_dir,
            fallback_hw=fallback_hw,
            args=args,
            image_index=i,
        )
        results.append(result)
        print('Saved:', result['sample_dir'])

    with open(out_dir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump({'results': results}, f, indent=2)

    print('Done. Summary written to {}'.format(str(out_dir / 'summary.json')))


if __name__ == '__main__':
    main()
