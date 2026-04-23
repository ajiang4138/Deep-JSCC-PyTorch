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

from constellation import pair_channels_to_symbols
from mic_mapper_visualizer import (build_model_from_checkpoint,
                                   infer_default_shape, load_input_tensor,
                                   tensor_image_to_numpy)


def compute_psnr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    mse = float(np.mean((x - y) ** 2))
    if mse <= eps:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def latent_to_iq_points(z_tensor: torch.Tensor, max_points: int = 50000) -> np.ndarray:
    points = pair_channels_to_symbols(z_tensor.detach().cpu()).reshape(-1, 2).numpy()
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
    return points


def plot_single_constellation_mrc(
    pre_pts: np.ndarray,
    post_pts: np.ndarray,
    codebook: Optional[np.ndarray],
    out_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(pre_pts[:, 0], pre_pts[:, 1], s=4, alpha=0.35, c='tab:blue', edgecolors='none')
    axes[0].set_title('Pre-MRC Symbols (Encoder Output)')
    axes[0].set_xlabel('I')
    axes[0].set_ylabel('Q')
    axes[0].grid(alpha=0.25)
    axes[0].axhline(0, color='k', linewidth=0.5, alpha=0.5)
    axes[0].axvline(0, color='k', linewidth=0.5, alpha=0.5)

    axes[1].scatter(post_pts[:, 0], post_pts[:, 1], s=4, alpha=0.35, c='tab:green', edgecolors='none', label='Post-MRC')
    if codebook is not None:
        axes[1].scatter(codebook[:, 0], codebook[:, 1], s=80, c='tab:red', marker='x', linewidths=2, label='MRC Codebook')
    axes[1].set_title('Post-MRC Symbols')
    axes[1].set_xlabel('I')
    axes[1].set_ylabel('Q')
    axes[1].grid(alpha=0.25)
    axes[1].axhline(0, color='k', linewidth=0.5, alpha=0.5)
    axes[1].axvline(0, color='k', linewidth=0.5, alpha=0.5)
    if codebook is not None:
        axes[1].legend(loc='best')

    for ax in axes:
        ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_triplet_plot(
    input_img: np.ndarray,
    no_mrc_img: np.ndarray,
    mrc_img: np.ndarray,
    psnr_no_mrc: float,
    psnr_mrc: float,
    out_path: Path,
):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(input_img)
    axes[0].set_title('Original')
    axes[0].axis('off')

    axes[1].imshow(no_mrc_img)
    axes[1].set_title('Output without MRC\nPSNR {:.2f} dB'.format(psnr_no_mrc))
    axes[1].axis('off')

    axes[2].imshow(mrc_img)
    axes[2].set_title('Output with MRC\nPSNR {:.2f} dB'.format(psnr_mrc))
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

        z_no_mrc = z_pre
        if hasattr(model, 'channel') and model.channel is not None:
            torch.manual_seed(args.seed + image_index)
            z_no_mrc = model.channel(z_no_mrc)
        x_hat_no_mrc = model.decoder(z_no_mrc)

        if model.mapper is None:
            raise RuntimeError('MRC mapper was not initialized')

        z_mrc, mrc_indices = model.mapper(z_pre, return_indices=True)
        if hasattr(model, 'channel') and model.channel is not None:
            torch.manual_seed(args.seed + image_index)
            z_mrc = model.channel(z_mrc)
        x_hat_mrc = model.decoder(z_mrc)

    x_np = tensor_image_to_numpy(x)
    no_np = tensor_image_to_numpy(x_hat_no_mrc)
    mrc_np = tensor_image_to_numpy(x_hat_mrc)
    pre_pts = latent_to_iq_points(z_pre)
    post_pts = latent_to_iq_points(z_mrc)
    codebook = model.mapper.get_effective_codebook().detach().cpu().numpy() if hasattr(model.mapper, 'get_effective_codebook') else None

    image_stem = Path(image_path).stem
    sample_dir = out_dir / image_stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(sample_dir / 'input_image.npy'), x.detach().cpu().numpy())
    np.save(str(sample_dir / 'x_hat_no_mrc.npy'), x_hat_no_mrc.detach().cpu().numpy())
    np.save(str(sample_dir / 'x_hat_mrc.npy'), x_hat_mrc.detach().cpu().numpy())
    np.save(str(sample_dir / 'z_pre_mapper.npy'), z_pre.detach().cpu().numpy())
    np.save(str(sample_dir / 'z_post_mapper.npy'), z_mrc.detach().cpu().numpy())
    np.save(str(sample_dir / 'z_pre_mapper_iq_points.npy'), pre_pts)
    np.save(str(sample_dir / 'z_post_mapper_iq_points.npy'), post_pts)
    np.save(str(sample_dir / 'mrc_mapper_indices.npy'), mrc_indices.detach().cpu().numpy())
    if codebook is not None:
        np.save(str(sample_dir / 'mrc_codebook.npy'), codebook)

    save_triplet_plot(
        input_img=x_np,
        no_mrc_img=no_np,
        mrc_img=mrc_np,
        psnr_no_mrc=compute_psnr(x_np, no_np),
        psnr_mrc=compute_psnr(x_np, mrc_np),
        out_path=sample_dir / 'triplet_original_no_mrc_mrc.png',
    )

    plot_single_constellation_mrc(
        pre_pts=pre_pts,
        post_pts=post_pts,
        codebook=codebook,
        out_path=sample_dir / 'constellation_no_mrc_vs_mrc.png',
    )

    metadata = {
        'input_source': input_source,
        'image_name': image_path,
        'checkpoint': args.checkpoint,
        'mrc_levels_per_axis': args.mrc_levels_per_axis,
        'mrc_init_bounds': args.mrc_init_bounds,
        'mrc_power_constraint_mode': args.mrc_power_constraint_mode,
        'channel': args.channel,
        'snr': args.snr,
        'saved_files': [
            'triplet_original_no_mrc_mrc.png',
            'constellation_no_mrc_vs_mrc.png',
            'input_image.npy',
            'x_hat_no_mrc.npy',
            'x_hat_mrc.npy',
            'z_pre_mapper.npy',
            'z_post_mapper.npy',
            'z_pre_mapper_iq_points.npy',
            'z_post_mapper_iq_points.npy',
            'mrc_mapper_indices.npy',
        ],
    }
    if codebook is not None:
        metadata['saved_files'].append('mrc_codebook.npy')

    with open(sample_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return {
        'image': image_path,
        'sample_dir': str(sample_dir),
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Generate MRC triplet and constellation outputs')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint used to reconstruct images')
    parser.add_argument('--images', nargs='+', type=str, required=True,
                        help='one or more input image paths')
    parser.add_argument('--output_dir', type=str, default='./out-mrc/test_mrc_triplet',
                        help='directory to save MRC outputs')

    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--inner_channel', type=int, default=None)
    parser.add_argument('--channel', type=str, default='AWGN', choices=['AWGN', 'Rayleigh'])
    parser.add_argument('--snr', type=float, default=7.0)

    parser.add_argument('--resize_h', type=int, default=0)
    parser.add_argument('--resize_w', type=int, default=0)

    parser.add_argument('--mrc_levels_per_axis', type=int, default=4)
    parser.add_argument('--mrc_init_bounds', type=str, default='')
    parser.add_argument('--mrc_power_constraint_mode', type=str, default='codebook', choices=['none', 'codebook', 'post_mapper'])

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

    model, meta, load_result = build_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device_obj,
        inner_channel_override=args.inner_channel,
        channel_override=args.channel,
        snr_override=args.snr,
    )

    mapper_kwargs = {
        'levels_per_axis': args.mrc_levels_per_axis,
        'init_bounds': args.mrc_init_bounds,
        'power_constraint_mode': args.mrc_power_constraint_mode,
    }
    model.set_mapper('mrc', mapper_kwargs)
    model.set_mapper_deploy_mode(True)

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
        json.dump({
            'results': results,
            'checkpoint_load_missing_keys': list(load_result.missing_keys),
            'checkpoint_load_unexpected_keys': list(load_result.unexpected_keys),
        }, f, indent=2)

    print('Done. Summary written to {}'.format(str(out_dir / 'summary.json')))


if __name__ == '__main__':
    main()