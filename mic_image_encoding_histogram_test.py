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

from constellation import (compute_codebook_usage_histogram,
                           map_to_mic_codebook, pair_channels_to_symbols)
from mic_mapper_visualizer import (build_model_from_checkpoint,
                                   infer_default_shape, load_input_tensor)


def estimate_codebook_from_latent(
    z_pre: torch.Tensor,
    k: int,
    seed: int,
    clip_value: float,
) -> np.ndarray:
    points = pair_channels_to_symbols(z_pre.detach().cpu()).reshape(-1, 2).numpy()
    points = np.clip(points, -clip_value, clip_value)

    rounded = np.round(points, decimals=5)
    unique = np.unique(rounded, axis=0)
    if 2 <= unique.shape[0] <= k:
        return unique

    rng = np.random.default_rng(seed)
    fit_points = points
    if fit_points.shape[0] > 30000:
        sample_idx = rng.choice(fit_points.shape[0], size=30000, replace=False)
        fit_points = fit_points[sample_idx]

    k = int(max(2, min(k, fit_points.shape[0])))
    init_idx = rng.choice(fit_points.shape[0], size=k, replace=False)
    centers = fit_points[init_idx].copy()
    labels = np.zeros(fit_points.shape[0], dtype=np.int64)

    for _ in range(30):
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


def latent_to_iq_points(z_tensor: torch.Tensor, max_points: int = 50000) -> np.ndarray:
    points = pair_channels_to_symbols(z_tensor.detach().cpu()).reshape(-1, 2).numpy()
    if max_points > 0 and points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[idx]
    return points


def plot_density_histogram(
    points: np.ndarray,
    out_path: Path,
    title: str,
    codebook: Optional[np.ndarray] = None,
    bins: int = 120,
):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    hist = ax.hist2d(points[:, 0], points[:, 1], bins=bins, cmap='magma')
    fig.colorbar(hist[3], ax=ax, label='count')

    if codebook is not None:
        ax.scatter(codebook[:, 0], codebook[:, 1], c='cyan', s=35, marker='x', linewidths=1.5, label='codebook')
        ax.legend(loc='upper right')

    ax.set_title(title)
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(alpha=0.18)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_usage_histogram(usage_counts: np.ndarray, out_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    indices = np.arange(usage_counts.shape[0])
    ax.bar(indices, usage_counts, color='tab:blue', width=0.85)
    ax.set_title(title)
    ax.set_xlabel('Codebook index')
    ax.set_ylabel('Assignments')
    ax.grid(axis='y', alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches='tight')
    plt.close(fig)


def save_metadata(sample_dir: Path, metadata: dict):
    with open(sample_dir / 'metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


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

        used_trained_mapper = bool(model.mapper is not None and args.prefer_trained_mapper)
        if used_trained_mapper:
            z_post = model.mapper(z_pre)
            codebook = model.mapper.get_effective_codebook().detach().cpu().numpy() if hasattr(model.mapper, 'get_effective_codebook') else None
            usage_counts = None
            mapper_indices = None
            if hasattr(model.mapper, 'last_stats') and isinstance(model.mapper.last_stats, dict):
                usage_obj = model.mapper.last_stats.get('usage_counts', None)
                if usage_obj is not None:
                    usage_counts = usage_obj.detach().cpu().numpy()
            if usage_counts is None and codebook is not None:
                post_full = pair_channels_to_symbols(z_post.detach().cpu()).reshape(-1, 2).numpy()
                d2 = ((post_full[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=-1)
                mapper_indices = np.argmin(d2, axis=1)
                usage_counts = np.bincount(mapper_indices, minlength=codebook.shape[0])
            mic_mode = 'trained_mapper'
        else:
            codebook = estimate_codebook_from_latent(
                z_pre=z_pre,
                k=args.mic_constellation_size,
                seed=args.seed + image_index,
                clip_value=args.mic_clip_value,
            )
            codebook_t = torch.tensor(codebook, dtype=z_pre.dtype, device=z_pre.device)
            z_post, mapper_indices_t = map_to_mic_codebook(
                z_tensor=z_pre,
                codebook=codebook_t,
                clip_value=args.mic_clip_value,
                power_constraint_mode=args.mic_power_constraint_mode,
            )
            mapper_indices = mapper_indices_t.detach().cpu().numpy()
            usage_counts = compute_codebook_usage_histogram(mapper_indices_t.reshape(-1), codebook.shape[0]).detach().cpu().numpy()
            mic_mode = 'estimated_codebook_mapper'

    pre_pts = latent_to_iq_points(z_pre, max_points=args.max_points)
    post_pts = latent_to_iq_points(z_post, max_points=args.max_points)

    image_stem = Path(image_path).stem
    sample_dir = out_dir / image_stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    np.save(str(sample_dir / 'input_image.npy'), x.detach().cpu().numpy())
    np.save(str(sample_dir / 'z_pre_mapper.npy'), z_pre.detach().cpu().numpy())
    np.save(str(sample_dir / 'z_post_mapper.npy'), z_post.detach().cpu().numpy())
    np.save(str(sample_dir / 'z_pre_mapper_iq_points.npy'), pre_pts)
    np.save(str(sample_dir / 'z_post_mapper_iq_points.npy'), post_pts)
    if mapper_indices is not None:
        np.save(str(sample_dir / 'mapper_indices.npy'), mapper_indices)
    if codebook is not None:
        np.save(str(sample_dir / 'mic_codebook_used.npy'), codebook)
    if usage_counts is not None:
        np.save(str(sample_dir / 'mic_usage_counts.npy'), usage_counts)

    plot_density_histogram(
        pre_pts,
        sample_dir / 'encoding_histogram_pre_mapper.png',
        title='Pre-MIC encoding histogram',
        codebook=None,
        bins=args.histogram_bins,
    )
    plot_density_histogram(
        post_pts,
        sample_dir / 'encoding_histogram_post_mapper.png',
        title='Post-MIC encoding histogram',
        codebook=codebook,
        bins=args.histogram_bins,
    )
    if usage_counts is not None:
        plot_usage_histogram(
            usage_counts,
            sample_dir / 'encoding_usage_histogram.png',
            title='MIC codebook usage histogram',
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    hist0 = axes[0].hist2d(pre_pts[:, 0], pre_pts[:, 1], bins=args.histogram_bins, cmap='magma')
    fig.colorbar(hist0[3], ax=axes[0], label='count')
    axes[0].set_title('Pre-MIC encoding')
    axes[0].set_xlabel('I')
    axes[0].set_ylabel('Q')
    axes[0].set_aspect('equal', adjustable='box')

    hist1 = axes[1].hist2d(post_pts[:, 0], post_pts[:, 1], bins=args.histogram_bins, cmap='magma')
    fig.colorbar(hist1[3], ax=axes[1], label='count')
    axes[1].set_title('Post-MIC encoding')
    axes[1].set_xlabel('I')
    axes[1].set_ylabel('Q')
    axes[1].set_aspect('equal', adjustable='box')
    if codebook is not None:
        axes[1].scatter(codebook[:, 0], codebook[:, 1], c='cyan', s=35, marker='x', linewidths=1.5)

    if usage_counts is not None:
        axes[2].bar(np.arange(usage_counts.shape[0]), usage_counts, color='tab:blue', width=0.85)
        axes[2].set_title('Codebook usage')
        axes[2].set_xlabel('Index')
        axes[2].set_ylabel('Assignments')
        axes[2].grid(axis='y', alpha=0.2)
    else:
        axes[2].axis('off')
        axes[2].text(0.5, 0.5, 'No mapper usage counts available', ha='center', va='center')

    fig.suptitle(f'{image_stem} - {mic_mode}', y=1.02)
    fig.tight_layout()
    fig.savefig(str(sample_dir / 'encoding_histograms.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)

    metadata = {
        'input_source': input_source,
        'image_name': image_path,
        'mic_mode': mic_mode,
        'checkpoint': args.checkpoint,
        'used_trained_mapper': used_trained_mapper,
        'has_mapper_in_model': bool(model.mapper is not None),
        'histogram_bins': args.histogram_bins,
        'mic_constellation_size': int(codebook.shape[0]) if codebook is not None else None,
        'mic_clip_value': args.mic_clip_value,
        'mic_power_constraint_mode': args.mic_power_constraint_mode,
        'saved_files': [
            'encoding_histograms.png',
            'encoding_histogram_pre_mapper.png',
            'encoding_histogram_post_mapper.png',
            'input_image.npy',
            'z_pre_mapper.npy',
            'z_post_mapper.npy',
            'z_pre_mapper_iq_points.npy',
            'z_post_mapper_iq_points.npy',
        ],
    }
    if usage_counts is not None:
        metadata['saved_files'].append('encoding_usage_histogram.png')
        metadata['saved_files'].append('mic_usage_counts.npy')
    if mapper_indices is not None:
        metadata['saved_files'].append('mapper_indices.npy')
    if codebook is not None:
        metadata['saved_files'].append('mic_codebook_used.npy')

    save_metadata(sample_dir, metadata)

    return {
        'image': image_path,
        'sample_dir': str(sample_dir),
        'mic_mode': mic_mode,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Generate encoding histograms for DeepJSCC/MIC images')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint used to reconstruct images')
    parser.add_argument('--images', nargs='+', type=str, required=True,
                        help='one or more input image paths')
    parser.add_argument('--output_dir', type=str, default='./out/test_mic_histograms',
                        help='directory to save histogram outputs')

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

    parser.add_argument('--histogram_bins', type=int, default=120)
    parser.add_argument('--max_points', type=int, default=50000)
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
    main()if __name__ == '__main__':
    main()