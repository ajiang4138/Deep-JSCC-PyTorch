#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import datasets, transforms

from constellation import map_to_mic_codebook, pair_channels_to_symbols
from mic_mapper_visualizer import (build_model_from_checkpoint,
                                   infer_default_shape, load_input_tensor,
                                   tensor_image_to_numpy)


def compute_psnr(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    mse = float(np.mean((x - y) ** 2))
    if mse <= eps:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def find_latest_cifar_checkpoint(root: Path) -> Path:
    candidates = sorted(root.glob('CIFAR10_*/epoch_*.pkl'))
    if not candidates:
        raise FileNotFoundError('No CIFAR-10 checkpoints found under {}'.format(str(root)))

    # Prefer newest modification time; break ties deterministically by path.
    return max(candidates, key=lambda p: (p.stat().st_mtime, str(p)))


def kmeans_codebook(points: np.ndarray, k: int, seed: int, max_iters: int = 30) -> np.ndarray:
    rng = np.random.default_rng(seed)
    k = int(max(2, min(k, points.shape[0])))

    init_idx = rng.choice(points.shape[0], size=k, replace=False)
    centers = points[init_idx].copy()
    labels = np.zeros(points.shape[0], dtype=np.int64)

    for _ in range(max_iters):
        d2 = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        new_centers = centers.copy()
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                new_centers[i] = points[mask].mean(axis=0)
            else:
                new_centers[i] = points[rng.integers(0, points.shape[0])]
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


def load_cifar10_sample(device: torch.device, sample_index: int) -> torch.Tensor:
    transform = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root='../dataset/', train=False, download=True, transform=transform)
    sample, _ = ds[int(sample_index) % len(ds)]
    return sample.unsqueeze(0).to(device)


def save_png(np_image: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img_u8 = np.clip(np_image * 255.0, 0.0, 255.0).astype(np.uint8)
    Image.fromarray(img_u8, mode='RGB').save(str(out_path))


def get_image_metadata(image_path: Path) -> Dict[str, object]:
    with Image.open(str(image_path)) as img:
        return {
            'path': str(image_path),
            'format': img.format,
            'mode': img.mode,
            'bands': list(img.getbands()),
            'size': list(img.size),
            'info_keys': sorted(list(img.info.keys())),
        }


def plot_rgb_hist(ax, image_np: np.ndarray, title: str):
    colors = ['r', 'g', 'b']
    labels = ['R', 'G', 'B']
    img_u8 = np.clip(image_np * 255.0, 0.0, 255.0).astype(np.uint8)
    for i in range(3):
        hist, bins = np.histogram(img_u8[:, :, i], bins=256, range=(0, 256))
        ax.bar(bins[:-1], hist, color=colors[i], label=labels[i], width=1.0, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Pixel Value')
    ax.set_ylabel('Count')
    ax.grid(alpha=0.25)
    ax.legend(loc='upper right')


def plot_metadata_and_histograms(
    input_np: np.ndarray,
    no_mic_np: np.ndarray,
    output_np: np.ndarray,
    input_meta: Dict[str, object],
    no_mic_meta: Dict[str, object],
    output_meta: Dict[str, object],
    psnr_no_mic: float,
    psnr_mic: float,
    out_path: Path,
):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    axes[0, 0].imshow(input_np)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(no_mic_np)
    axes[0, 1].set_title('Output w/o MIC\nPSNR {:.2f} dB'.format(psnr_no_mic))
    axes[0, 1].axis('off')

    axes[0, 2].imshow(output_np)
    axes[0, 2].set_title('Output w/ MIC\nPSNR {:.2f} dB'.format(psnr_mic))
    axes[0, 2].axis('off')

    plot_rgb_hist(axes[1, 0], input_np, 'Input RGB Histogram')
    plot_rgb_hist(axes[1, 1], no_mic_np, 'No-MIC RGB Histogram')
    plot_rgb_hist(axes[1, 2], output_np, 'MIC Output RGB Histogram')

    mode_text = (
        'Encoding Metadata\n'
        'Input mode: {} ({})\n'
        'No-MIC mode: {} ({})\n'
        'Output mode: {} ({})\n'
        'Input format: {}\n'
        'No-MIC format: {}\n'
        'Output format: {}'
    ).format(
        input_meta.get('mode', 'unknown'), ','.join(input_meta.get('bands', [])),
        no_mic_meta.get('mode', 'unknown'), ','.join(no_mic_meta.get('bands', [])),
        output_meta.get('mode', 'unknown'), ','.join(output_meta.get('bands', [])),
        input_meta.get('format', 'unknown'),
        no_mic_meta.get('format', 'unknown'),
        output_meta.get('format', 'unknown'),
    )
    fig.text(0.5, 0.02, mode_text, ha='center', va='bottom', fontsize=10)

    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run MIC output generation and metadata histogram for CIFAR-10 sample'
    )
    parser.add_argument('--checkpoint', type=str, default='',
                        help='optional checkpoint path; if empty, use latest CIFAR-10 checkpoint')
    parser.add_argument('--checkpoint_root', type=str, default='./out/checkpoints',
                        help='root directory for auto-discovering CIFAR-10 checkpoints')
    parser.add_argument('--output_dir', type=str, default='./out/test_cifar10_snr15_metadata_hist',
                        help='output directory')
    parser.add_argument('--snr', type=float, default=15.0,
                        help='SNR override for rerun')
    parser.add_argument('--channel', type=str, default=None, choices=['AWGN', 'Rayleigh'],
                        help='optional channel override')
    parser.add_argument('--inner_channel', type=int, default=None,
                        help='optional latent channel override')
    parser.add_argument('--sample_index', type=int, default=0,
                        help='CIFAR-10 test sample index')
    parser.add_argument('--input_image', type=str, default='',
                        help='optional image path; when set, bypass CIFAR-10 sampling')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--mic_mapper_source', type=str, default='auto',
                        choices=['auto', 'trained', 'estimated'],
                        help='MIC mapper source: auto (trained if available, else estimated), trained, or estimated')
    parser.add_argument('--prefer_trained_mapper', action='store_true',
                        help='deprecated compatibility flag; equivalent to --mic_mapper_source trained when source is auto')
    parser.add_argument('--mic_constellation_size', type=int, default=16,
                        help='constellation size for estimated MIC mapping when mapper is absent')
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
    device = torch.device(args.device)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_latest_cifar_checkpoint(Path(args.checkpoint_root))
    print('Using checkpoint: {}'.format(str(checkpoint_path)))

    model, meta, load_result = build_model_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        device=device,
        inner_channel_override=args.inner_channel,
        channel_override=args.channel,
        snr_override=args.snr,
    )
    model.eval()

    if args.input_image:
        fallback_hw = infer_default_shape(meta)
        x, _ = load_input_tensor(
            image_path=args.input_image,
            device=device,
            resize_hw=None,
            fallback_hw=fallback_hw,
        )
    else:
        x = load_cifar10_sample(device=device, sample_index=args.sample_index)

    with torch.no_grad():
        z_pre = model.encoder(x)

        z_no_mic = z_pre
        if hasattr(model, 'channel') and model.channel is not None:
            torch.manual_seed(args.seed)
            z_no_mic = model.channel(z_no_mic)
        x_hat_no_mic = model.decoder(z_no_mic)

        mapper_source = args.mic_mapper_source
        if mapper_source == 'auto' and args.prefer_trained_mapper:
            mapper_source = 'trained'

        has_trained_mapper = bool(model.mapper is not None)
        if mapper_source == 'trained' and not has_trained_mapper:
            raise ValueError('Requested trained MIC mapper, but checkpoint has no mapper. Use --mic_mapper_source estimated or auto.')

        use_trained_mapper = bool(
            has_trained_mapper and (mapper_source == 'trained' or mapper_source == 'auto')
        )
        if use_trained_mapper:
            z_mic = model.mapper(z_pre)
            codebook = model.mapper.get_effective_codebook().detach().cpu().numpy() if hasattr(model.mapper, 'get_effective_codebook') else None
            mic_mode = 'trained_mapper'
        else:
            codebook = estimate_codebook_from_latent(
                z_pre=z_pre,
                k=args.mic_constellation_size,
                seed=args.seed,
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
            torch.manual_seed(args.seed)
            z_mic = model.channel(z_mic)

        x_hat_mic = model.decoder(z_mic)

    input_np = tensor_image_to_numpy(x)
    no_mic_np = tensor_image_to_numpy(x_hat_no_mic)
    output_np = tensor_image_to_numpy(x_hat_mic)
    psnr_no_mic = compute_psnr(input_np, no_mic_np)
    psnr_mic = compute_psnr(input_np, output_np)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_png = out_dir / 'input_image.png'
    no_mic_png = out_dir / 'output_image_no_mic.png'
    output_png = out_dir / 'output_image_mic.png'
    save_png(input_np, input_png)
    save_png(no_mic_np, no_mic_png)
    save_png(output_np, output_png)

    np.save(str(out_dir / 'input_image.npy'), x.detach().cpu().numpy())
    np.save(str(out_dir / 'output_image_no_mic.npy'), x_hat_no_mic.detach().cpu().numpy())
    np.save(str(out_dir / 'output_image_mic.npy'), x_hat_mic.detach().cpu().numpy())
    if codebook is not None:
        np.save(str(out_dir / 'mic_codebook_used.npy'), codebook)

    input_meta = get_image_metadata(input_png)
    no_mic_meta = get_image_metadata(no_mic_png)
    output_meta = get_image_metadata(output_png)

    plot_metadata_and_histograms(
        input_np=input_np,
        no_mic_np=no_mic_np,
        output_np=output_np,
        input_meta=input_meta,
        no_mic_meta=no_mic_meta,
        output_meta=output_meta,
        psnr_no_mic=psnr_no_mic,
        psnr_mic=psnr_mic,
        out_path=out_dir / 'metadata_histogram.png',
    )

    summary = {
        'checkpoint_used': str(checkpoint_path),
        'snr_used': float(args.snr),
        'sample_index': int(args.sample_index),
        'mic_mode': mic_mode,
        'mic_mapper_source_requested': mapper_source,
        'has_mapper_in_model': has_trained_mapper,
        'used_trained_mapper': use_trained_mapper,
        'psnr_no_mic_db': psnr_no_mic,
        'psnr_mic_db': psnr_mic,
        'input_metadata': input_meta,
        'output_no_mic_metadata': no_mic_meta,
        'output_metadata': output_meta,
        'load_missing_keys': list(load_result.missing_keys),
        'load_unexpected_keys': list(load_result.unexpected_keys),
        'saved_files': [
            'input_image.png',
            'output_image_no_mic.png',
            'output_image_mic.png',
            'metadata_histogram.png',
            'input_image.npy',
            'output_image_no_mic.npy',
            'output_image_mic.npy',
        ],
    }
    if codebook is not None:
        summary['saved_files'].append('mic_codebook_used.npy')

    with open(out_dir / 'metadata_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print('Saved artifacts to {}'.format(str(out_dir)))


if __name__ == '__main__':
    main()
