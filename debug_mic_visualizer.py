#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Debug visualization utility for DeepJSCC + MIC.

Features:
- Load checkpoint and run encoder -> mapper -> channel -> decoder.
- Extract z_pre_mapper, z_post_mapper, x_hat via model.forward_debug().
- Plot I/Q constellations before and after MIC (adjacent channel pairing).
- Overlay learned MIC codebook on post-mapper constellation.
- Save side-by-side input vs reconstruction image.
- Save .npy arrays for symbols, reconstructions, mapper indices and codebook.
- Optional baseline vs MIC comparison on the same input image.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from constellation import pair_channels_to_symbols
from model import DeepJSCC


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k.replace('module.', ''): v for k, v in state_dict.items()}


def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and 'model_state_dict' in checkpoint_obj:
        return checkpoint_obj['model_state_dict'], checkpoint_obj
    return checkpoint_obj, {}


def parse_c_from_checkpoint_path(checkpoint_path: str) -> Optional[int]:
    parts = Path(checkpoint_path).parent.name.split('_')
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def infer_default_shape(meta: Dict[str, object]) -> Tuple[int, int]:
    params = meta.get('params', {}) if isinstance(meta, dict) else {}
    dataset_name = str(params.get('dataset', '')).lower()
    if dataset_name == 'cifar10':
        return 32, 32
    if dataset_name == 'imagenet':
        return 128, 128
    return 128, 128


def infer_mapper_config(state_dict: Dict[str, torch.Tensor], meta: Dict[str, object]) -> Tuple[str, Optional[Dict[str, object]]]:
    mapper_cfg = {}
    if isinstance(meta, dict):
        mapper_cfg = dict(meta.get('mapper_config', {}) or {})

    has_mapper_weight = any(k.startswith('mapper.') for k in state_dict.keys())
    mapper_type = str(mapper_cfg.get('mapper_type', 'mic' if has_mapper_weight else 'none')).lower()

    if mapper_type == 'none':
        return 'none', None

    codebook_key = 'mapper.codebook'
    codebook = state_dict.get(codebook_key, None)
    constellation_size = int(mapper_cfg.get('constellation_size', codebook.shape[0] if codebook is not None else 16))

    mapper_kwargs = {
        'constellation_size': constellation_size,
        'clip_value': float(mapper_cfg.get('clip_value', 2.0)),
        'temperature': float(mapper_cfg.get('temperature', 0.1)),
        'delta': mapper_cfg.get('delta', None),
        'hard_forward': bool(mapper_cfg.get('hard_forward', True)),
        'train_mode': str(mapper_cfg.get('train_mode', 'hard_forward_soft_backward')),
        'power_constraint_mode': str(mapper_cfg.get('power_constraint_mode', 'codebook')),
    }
    return 'mic', mapper_kwargs


def build_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    inner_channel_override: Optional[int],
    channel_override: Optional[str],
    snr_override: Optional[float],
):
    checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    state_dict_raw, meta = extract_state_dict(checkpoint_obj)
    state_dict = strip_module_prefix(state_dict_raw)

    c = inner_channel_override
    if c is None and isinstance(meta, dict):
        c = meta.get('inner_channel', None)
    if c is None:
        c = parse_c_from_checkpoint_path(checkpoint_path)
    if c is None:
        raise ValueError('Cannot infer inner channel c. Please pass --inner_channel.')

    params = meta.get('params', {}) if isinstance(meta, dict) else {}
    channel_type = channel_override if channel_override is not None else params.get('channel', 'AWGN')
    snr = snr_override if snr_override is not None else params.get('snr', None)

    mapper_type, mapper_kwargs = infer_mapper_config(state_dict, meta)

    model = DeepJSCC(
        c=int(c),
        channel_type=str(channel_type),
        snr=snr,
        mapper_type=mapper_type,
        mapper_kwargs=mapper_kwargs,
    ).to(device)

    load_result = model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, meta, load_result


def load_input_tensor(
    image_path: str,
    device: torch.device,
    resize_hw: Optional[Tuple[int, int]],
    fallback_hw: Tuple[int, int],
):
    if image_path and os.path.exists(image_path):
        image = Image.open(image_path).convert('RGB')
        tfms = []
        if resize_hw is not None:
            tfms.append(transforms.Resize(resize_hw))
        tfms.append(transforms.ToTensor())
        x = transforms.Compose(tfms)(image).unsqueeze(0).to(device)
        input_source = image_path
    else:
        h, w = fallback_hw
        x = torch.rand(1, 3, h, w, device=device)
        input_source = 'random'
    return x, input_source


def latent_to_iq_points(z_tensor: torch.Tensor, max_points: int = 50000) -> np.ndarray:
    symbols = pair_channels_to_symbols(z_tensor.detach().cpu())
    points = symbols.reshape(-1, 2).numpy()

    if max_points > 0 and points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[idx]

    return points


def tensor_image_to_numpy(x: torch.Tensor) -> np.ndarray:
    img = x.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(img, 0.0, 1.0)


def save_npy(path: Path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), array)


def build_decision_regions(codebook: np.ndarray, xlim: Tuple[float, float], ylim: Tuple[float, float], grid_size: int = 320):
    xs = np.linspace(xlim[0], xlim[1], grid_size)
    ys = np.linspace(ylim[0], ylim[1], grid_size)
    xx, yy = np.meshgrid(xs, ys)
    grid_points = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=-1)

    d2 = ((grid_points[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=-1)
    labels = np.argmin(d2, axis=1).reshape(xx.shape)
    return xx, yy, labels


def plot_region_boundaries(ax, decision_centers: Optional[np.ndarray], post_pts: np.ndarray):
    if decision_centers is None or len(decision_centers) < 2:
        return

    xmin = min(post_pts[:, 0].min(), decision_centers[:, 0].min())
    xmax = max(post_pts[:, 0].max(), decision_centers[:, 0].max())
    ymin = min(post_pts[:, 1].min(), decision_centers[:, 1].min())
    ymax = max(post_pts[:, 1].max(), decision_centers[:, 1].max())

    pad_x = max((xmax - xmin) * 0.08, 1e-3)
    pad_y = max((ymax - ymin) * 0.08, 1e-3)
    xlim = (xmin - pad_x, xmax + pad_x)
    ylim = (ymin - pad_y, ymax + pad_y)

    xx, yy, labels = build_decision_regions(decision_centers, xlim=xlim, ylim=ylim)
    levels = np.arange(0.5, decision_centers.shape[0] - 0.5, 1.0)
    if len(levels) > 0:
        ax.contour(xx, yy, labels, levels=levels, colors='k', linewidths=0.7, alpha=0.35)


def _kmeans_centroids(points: np.ndarray, k: int, seed: int = 0, max_iters: int = 25):
    if points.shape[0] == 0:
        return None

    k = int(max(1, min(k, points.shape[0])))
    rng = np.random.default_rng(seed)

    init_idx = rng.choice(points.shape[0], size=k, replace=False)
    centroids = points[init_idx].copy()
    labels = np.zeros(points.shape[0], dtype=np.int64)

    for _ in range(max_iters):
        d2 = ((points[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
        new_labels = np.argmin(d2, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        new_centroids = centroids.copy()
        for i in range(k):
            mask = labels == i
            if np.any(mask):
                new_centroids[i] = points[mask].mean(axis=0)
            else:
                new_centroids[i] = points[rng.integers(0, points.shape[0])]
        centroids = new_centroids

    return centroids, labels


def _estimate_centers_from_post_symbols(post_symbols: np.ndarray, target_k: int, seed: int = 0):
    if post_symbols.shape[0] == 0:
        return None

    rounded = np.round(post_symbols, decimals=5)
    unique_points = np.unique(rounded, axis=0)

    if unique_points.shape[0] >= 2 and unique_points.shape[0] <= target_k:
        d2 = ((post_symbols[:, None, :] - unique_points[None, :, :]) ** 2).sum(axis=-1)
        labels = np.argmin(d2, axis=1)
        return unique_points, labels

    fit_points = post_symbols
    if fit_points.shape[0] > 30000:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(fit_points.shape[0], size=30000, replace=False)
        fit_points = fit_points[sample_idx]

    kmeans_result = _kmeans_centroids(fit_points, k=target_k, seed=seed)
    if kmeans_result is None:
        return None
    centers, _ = kmeans_result

    d2_full = ((post_symbols[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
    labels_full = np.argmin(d2_full, axis=1)
    return centers, labels_full


def compute_cluster_centroids(
    post_symbols: np.ndarray,
    mapper_indices: Optional[np.ndarray],
    codebook: Optional[np.ndarray],
    fallback_k: int,
    seed: int,
):
    if mapper_indices is None:
        if codebook is not None:
            idx = np.argmin(((post_symbols[:, None, :] - codebook[None, :, :]) ** 2).sum(axis=-1), axis=1)
        else:
            estimated = _estimate_centers_from_post_symbols(post_symbols, target_k=max(2, fallback_k), seed=seed)
            if estimated is None:
                return None
            centers, idx = estimated
            centroid_labels = list(range(centers.shape[0]))
            return centers, centroid_labels
    else:
        idx = mapper_indices.reshape(-1)

    centroids = []
    labels = []
    for k in np.unique(idx):
        pts = post_symbols[idx == k]
        if pts.size == 0:
            continue
        centroids.append(pts.mean(axis=0))
        labels.append(int(k))

    if not centroids:
        return None
    return np.asarray(centroids), labels


def plot_single_constellation(
    pre_pts: np.ndarray,
    post_pts: np.ndarray,
    codebook: Optional[np.ndarray],
    centroid_info,
    out_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(pre_pts[:, 0], pre_pts[:, 1], s=4, alpha=0.35, c='tab:blue', edgecolors='none')
    axes[0].set_title('Pre-MIC Symbols (Encoder Output)')
    axes[0].set_xlabel('I')
    axes[0].set_ylabel('Q')
    axes[0].grid(alpha=0.25)
    axes[0].axhline(0, color='k', linewidth=0.5, alpha=0.5)
    axes[0].axvline(0, color='k', linewidth=0.5, alpha=0.5)

    axes[1].scatter(post_pts[:, 0], post_pts[:, 1], s=4, alpha=0.35, c='tab:green', edgecolors='none', label='Post-MIC')
    decision_centers = codebook
    if decision_centers is None and centroid_info is not None:
        decision_centers = centroid_info[0]
    plot_region_boundaries(axes[1], decision_centers=decision_centers, post_pts=post_pts)
    if codebook is not None:
        axes[1].scatter(codebook[:, 0], codebook[:, 1], s=80, c='tab:red', marker='x', linewidths=2, label='MIC Codebook')
    if centroid_info is not None:
        centroids, centroid_labels = centroid_info
        axes[1].scatter(centroids[:, 0], centroids[:, 1], s=56, c='tab:orange', marker='o', edgecolors='k', linewidths=0.6, label='Cluster Centroids')
        for i in range(len(centroid_labels)):
            axes[1].text(centroids[i, 0], centroids[i, 1], str(centroid_labels[i]), fontsize=7, alpha=0.85)
    axes[1].set_title('Post-MIC Symbols')
    axes[1].set_xlabel('I')
    axes[1].set_ylabel('Q')
    axes[1].grid(alpha=0.25)
    axes[1].axhline(0, color='k', linewidth=0.5, alpha=0.5)
    axes[1].axvline(0, color='k', linewidth=0.5, alpha=0.5)
    if codebook is not None or centroid_info is not None:
        axes[1].legend(loc='best')

    for ax in axes:
        ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_single_reconstruction(input_img: np.ndarray, recon_img: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(input_img)
    axes[0].set_title('Input')
    axes[0].axis('off')

    axes[1].imshow(recon_img)
    axes[1].set_title('Reconstruction')
    axes[1].axis('off')

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_constellation(
    baseline_pre: np.ndarray,
    mic_pre: np.ndarray,
    mic_post: np.ndarray,
    mic_codebook: Optional[np.ndarray],
    mic_centroid_info,
    out_path: Path,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].scatter(baseline_pre[:, 0], baseline_pre[:, 1], s=4, alpha=0.35, c='tab:blue', edgecolors='none')
    axes[0].set_title('Baseline: Encoder Symbols')

    axes[1].scatter(mic_pre[:, 0], mic_pre[:, 1], s=4, alpha=0.35, c='tab:purple', edgecolors='none')
    axes[1].set_title('MIC: Pre-Mapper Symbols')

    axes[2].scatter(mic_post[:, 0], mic_post[:, 1], s=4, alpha=0.35, c='tab:green', edgecolors='none', label='Post-MIC')
    decision_centers = mic_codebook
    if decision_centers is None and mic_centroid_info is not None:
        decision_centers = mic_centroid_info[0]
    plot_region_boundaries(axes[2], decision_centers=decision_centers, post_pts=mic_post)
    if mic_codebook is not None:
        axes[2].scatter(mic_codebook[:, 0], mic_codebook[:, 1], s=80, c='tab:red', marker='x', linewidths=2, label='MIC Codebook')
    if mic_centroid_info is not None:
        centroids, centroid_labels = mic_centroid_info
        axes[2].scatter(centroids[:, 0], centroids[:, 1], s=56, c='tab:orange', marker='o', edgecolors='k', linewidths=0.6, label='Cluster Centroids')
        for i in range(len(centroid_labels)):
            axes[2].text(centroids[i, 0], centroids[i, 1], str(centroid_labels[i]), fontsize=7, alpha=0.85)
    axes[2].set_title('MIC: Post-Mapper Symbols')

    for ax in axes:
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.grid(alpha=0.25)
        ax.axhline(0, color='k', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='k', linewidth=0.5, alpha=0.5)
        ax.set_aspect('equal', adjustable='box')

    if mic_codebook is not None or mic_centroid_info is not None:
        axes[2].legend(loc='best')

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches='tight')
    plt.close(fig)


def plot_comparison_reconstruction(input_img: np.ndarray, baseline_img: np.ndarray, mic_img: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(input_img)
    axes[0].set_title('Input')
    axes[0].axis('off')

    axes[1].imshow(baseline_img)
    axes[1].set_title('Baseline Reconstruction')
    axes[1].axis('off')

    axes[2].imshow(mic_img)
    axes[2].set_title('MIC Reconstruction')
    axes[2].axis('off')

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches='tight')
    plt.close(fig)


def run_single(args, device: torch.device):
    model, meta, load_result = build_model_from_checkpoint(
        checkpoint_path=args.checkpoint,
        device=device,
        inner_channel_override=args.inner_channel,
        channel_override=args.channel,
        snr_override=args.snr,
    )

    fallback_hw = infer_default_shape(meta)
    resize_hw = (args.resize_h, args.resize_w) if args.resize_h > 0 and args.resize_w > 0 else None
    x, input_source = load_input_tensor(args.input_image, device, resize_hw, fallback_hw)

    with torch.no_grad():
        debug_out = model.forward_debug(x, return_mapper_indices=True)

    z_pre = debug_out['z_pre_mapper']
    z_post = debug_out['z_post_mapper']
    x_hat = debug_out['x_hat']
    mapper_indices = debug_out.get('mapper_indices', None)

    codebook = None
    if model.mapper is not None and hasattr(model.mapper, 'get_effective_codebook'):
        codebook = model.mapper.get_effective_codebook().detach().cpu().numpy()

    pre_pts = latent_to_iq_points(z_pre, max_points=args.max_points)
    post_pts = latent_to_iq_points(z_post, max_points=args.max_points)
    post_symbols_full = pair_channels_to_symbols(z_post.detach().cpu()).reshape(-1, 2).numpy()
    centroid_info = compute_cluster_centroids(
        post_symbols_full,
        mapper_indices.detach().cpu().numpy() if mapper_indices is not None else None,
        codebook,
        fallback_k=args.overlay_k,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_npy(out_dir / 'input_image.npy', x.detach().cpu().numpy())
    save_npy(out_dir / 'x_hat.npy', x_hat.detach().cpu().numpy())
    save_npy(out_dir / 'z_pre_mapper.npy', z_pre.detach().cpu().numpy())
    save_npy(out_dir / 'z_post_mapper.npy', z_post.detach().cpu().numpy())
    save_npy(out_dir / 'z_pre_mapper_iq_points.npy', pre_pts)
    save_npy(out_dir / 'z_post_mapper_iq_points.npy', post_pts)

    if mapper_indices is not None:
        save_npy(out_dir / 'mapper_indices.npy', mapper_indices.detach().cpu().numpy())
    if codebook is not None:
        save_npy(out_dir / 'mic_codebook.npy', codebook)

    plot_single_constellation(pre_pts, post_pts, codebook, centroid_info, out_dir / 'constellation_pre_post.png')
    plot_single_reconstruction(
        tensor_image_to_numpy(x),
        tensor_image_to_numpy(x_hat),
        out_dir / 'input_vs_reconstruction.png',
    )

    metadata = {
        'mode': 'single',
        'checkpoint': args.checkpoint,
        'input_source': input_source,
        'device': str(device),
        'load_missing_keys': list(load_result.missing_keys),
        'load_unexpected_keys': list(load_result.unexpected_keys),
        'has_mapper': model.mapper is not None,
        'saved_files': [
            'constellation_pre_post.png',
            'input_vs_reconstruction.png',
            'input_image.npy',
            'x_hat.npy',
            'z_pre_mapper.npy',
            'z_post_mapper.npy',
            'z_pre_mapper_iq_points.npy',
            'z_post_mapper_iq_points.npy',
        ],
    }
    if mapper_indices is not None:
        metadata['saved_files'].append('mapper_indices.npy')
    if codebook is not None:
        metadata['saved_files'].append('mic_codebook.npy')

    with open(out_dir / 'debug_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print('Saved debug outputs to {}'.format(str(out_dir)))


def run_compare(args, device: torch.device):
    baseline_model, baseline_meta, baseline_load_result = build_model_from_checkpoint(
        checkpoint_path=args.baseline_checkpoint,
        device=device,
        inner_channel_override=args.inner_channel,
        channel_override=args.channel,
        snr_override=args.snr,
    )
    mic_model, mic_meta, mic_load_result = build_model_from_checkpoint(
        checkpoint_path=args.mic_checkpoint,
        device=device,
        inner_channel_override=args.inner_channel,
        channel_override=args.channel,
        snr_override=args.snr,
    )

    fallback_hw = infer_default_shape(mic_meta if mic_meta else baseline_meta)
    resize_hw = (args.resize_h, args.resize_w) if args.resize_h > 0 and args.resize_w > 0 else None
    x, input_source = load_input_tensor(args.input_image, device, resize_hw, fallback_hw)

    with torch.no_grad():
        baseline_out = baseline_model.forward_debug(x, return_mapper_indices=False)
        mic_out = mic_model.forward_debug(x, return_mapper_indices=True)

    baseline_pre = baseline_out['z_pre_mapper']
    baseline_xhat = baseline_out['x_hat']

    mic_pre = mic_out['z_pre_mapper']
    mic_post = mic_out['z_post_mapper']
    mic_xhat = mic_out['x_hat']
    mic_indices = mic_out.get('mapper_indices', None)

    mic_codebook = None
    if mic_model.mapper is not None and hasattr(mic_model.mapper, 'get_effective_codebook'):
        mic_codebook = mic_model.mapper.get_effective_codebook().detach().cpu().numpy()

    baseline_pre_pts = latent_to_iq_points(baseline_pre, max_points=args.max_points)
    mic_pre_pts = latent_to_iq_points(mic_pre, max_points=args.max_points)
    mic_post_pts = latent_to_iq_points(mic_post, max_points=args.max_points)
    mic_post_full = pair_channels_to_symbols(mic_post.detach().cpu()).reshape(-1, 2).numpy()
    mic_centroid_info = compute_cluster_centroids(
        mic_post_full,
        mic_indices.detach().cpu().numpy() if mic_indices is not None else None,
        mic_codebook,
        fallback_k=args.overlay_k,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_npy(out_dir / 'input_image.npy', x.detach().cpu().numpy())

    save_npy(out_dir / 'baseline_z_pre_mapper.npy', baseline_pre.detach().cpu().numpy())
    save_npy(out_dir / 'baseline_x_hat.npy', baseline_xhat.detach().cpu().numpy())
    save_npy(out_dir / 'baseline_z_pre_mapper_iq_points.npy', baseline_pre_pts)

    save_npy(out_dir / 'mic_z_pre_mapper.npy', mic_pre.detach().cpu().numpy())
    save_npy(out_dir / 'mic_z_post_mapper.npy', mic_post.detach().cpu().numpy())
    save_npy(out_dir / 'mic_x_hat.npy', mic_xhat.detach().cpu().numpy())
    save_npy(out_dir / 'mic_z_pre_mapper_iq_points.npy', mic_pre_pts)
    save_npy(out_dir / 'mic_z_post_mapper_iq_points.npy', mic_post_pts)

    if mic_indices is not None:
        save_npy(out_dir / 'mic_mapper_indices.npy', mic_indices.detach().cpu().numpy())
    if mic_codebook is not None:
        save_npy(out_dir / 'mic_codebook.npy', mic_codebook)

    plot_comparison_constellation(
        baseline_pre_pts,
        mic_pre_pts,
        mic_post_pts,
        mic_codebook,
        mic_centroid_info,
        out_dir / 'constellation_baseline_vs_mic.png',
    )
    plot_comparison_reconstruction(
        tensor_image_to_numpy(x),
        tensor_image_to_numpy(baseline_xhat),
        tensor_image_to_numpy(mic_xhat),
        out_dir / 'reconstruction_baseline_vs_mic.png',
    )

    metadata = {
        'mode': 'compare',
        'baseline_checkpoint': args.baseline_checkpoint,
        'mic_checkpoint': args.mic_checkpoint,
        'input_source': input_source,
        'device': str(device),
        'baseline_load_missing_keys': list(baseline_load_result.missing_keys),
        'baseline_load_unexpected_keys': list(baseline_load_result.unexpected_keys),
        'mic_load_missing_keys': list(mic_load_result.missing_keys),
        'mic_load_unexpected_keys': list(mic_load_result.unexpected_keys),
        'saved_files': [
            'constellation_baseline_vs_mic.png',
            'reconstruction_baseline_vs_mic.png',
            'input_image.npy',
            'baseline_z_pre_mapper.npy',
            'baseline_x_hat.npy',
            'baseline_z_pre_mapper_iq_points.npy',
            'mic_z_pre_mapper.npy',
            'mic_z_post_mapper.npy',
            'mic_x_hat.npy',
            'mic_z_pre_mapper_iq_points.npy',
            'mic_z_post_mapper_iq_points.npy',
        ],
    }
    if mic_indices is not None:
        metadata['saved_files'].append('mic_mapper_indices.npy')
    if mic_codebook is not None:
        metadata['saved_files'].append('mic_codebook.npy')

    with open(out_dir / 'debug_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print('Saved comparison outputs to {}'.format(str(out_dir)))


def parse_args():
    parser = argparse.ArgumentParser(description='DeepJSCC MIC debug visualizer')

    parser.add_argument('--checkpoint', default='', type=str,
                        help='single-checkpoint mode path')
    parser.add_argument('--baseline_checkpoint', default='', type=str,
                        help='baseline checkpoint for comparison mode')
    parser.add_argument('--mic_checkpoint', default='', type=str,
                        help='MIC checkpoint for comparison mode')

    parser.add_argument('--input_image', default='', type=str,
                        help='optional input image path')
    parser.add_argument('--output_dir', default='./out/debug_mic_visualizer', type=str,
                        help='output directory for images and numpy arrays')
    parser.add_argument('--device', default='cpu', type=str,
                        choices=['cpu', 'cuda'])

    parser.add_argument('--inner_channel', default=None, type=int,
                        help='optional override for latent c')
    parser.add_argument('--channel', default=None, type=str,
                        choices=['AWGN', 'Rayleigh'],
                        help='optional channel override')
    parser.add_argument('--snr', default=None, type=float,
                        help='optional SNR override')

    parser.add_argument('--resize_h', default=0, type=int,
                        help='optional input resize height')
    parser.add_argument('--resize_w', default=0, type=int,
                        help='optional input resize width')

    parser.add_argument('--max_points', default=50000, type=int,
                        help='maximum I/Q points to scatter in constellation plots')
    parser.add_argument('--overlay_k', default=16, type=int,
                        help='number of overlay decision centers when codebook is unavailable')
    parser.add_argument('--seed', default=42, type=int,
                        help='random seed (for point subsampling)')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, switching to CPU.')
        args.device = 'cpu'

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)

    compare_mode = bool(args.baseline_checkpoint and args.mic_checkpoint)
    single_mode = bool(args.checkpoint)

    if compare_mode:
        run_compare(args, device)
    elif single_mode:
        run_single(args, device)
    else:
        raise ValueError('Specify either --checkpoint for single mode OR both --baseline_checkpoint and --mic_checkpoint for comparison mode.')


if __name__ == '__main__':
    main()
