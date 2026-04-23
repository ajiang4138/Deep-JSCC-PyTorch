import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def _parse_init_bounds(init_bounds: Optional[object]) -> Tuple[float, float]:
    if init_bounds is None:
        return -1.0, 1.0

    if isinstance(init_bounds, (tuple, list)) and len(init_bounds) == 2:
        low = float(init_bounds[0])
        high = float(init_bounds[1])
    else:
        raw = str(init_bounds).strip()
        if not raw:
            return -1.0, 1.0
        parts = [p.strip() for p in raw.split(',')]
        if len(parts) != 2:
            raise ValueError("init_bounds must be empty or in 'low,high' format")
        low = float(parts[0])
        high = float(parts[1])

    if not (low < high):
        raise ValueError("init_bounds must satisfy low < high")
    return low, high


def _build_regular_codebook(levels_per_axis: int, init_bounds: Tuple[float, float]) -> torch.Tensor:
    low, high = init_bounds
    levels = torch.linspace(low, high, levels_per_axis)
    grid_i, grid_q = torch.meshgrid(levels, levels, indexing="ij")
    codebook = torch.stack([grid_i.reshape(-1), grid_q.reshape(-1)], dim=-1)
    return normalize_constellation_power(codebook)


def _ensure_batched_latent(z_tensor: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    if z_tensor.dim() == 3:
        return z_tensor.unsqueeze(0), True
    if z_tensor.dim() == 4:
        return z_tensor, False
    raise ValueError("Expected latent tensor with 3D or 4D shape")


def pair_channels_to_symbols(z_tensor: torch.Tensor) -> torch.Tensor:
    """Convert [B, 2*c, H, W] into [B, c, H, W, 2] using adjacent channel pairs."""
    z_tensor, _ = _ensure_batched_latent(z_tensor)
    b, channels, h, w = z_tensor.shape
    if channels % 2 != 0:
        raise ValueError("Channel dimension must be even to form I/Q pairs")

    c = channels // 2
    return z_tensor.contiguous().view(b, c, 2, h, w).permute(0, 1, 3, 4, 2).contiguous()


def unpair_symbols_to_channels(symbol_tensor: torch.Tensor) -> torch.Tensor:
    """Convert [B, c, H, W, 2] into [B, 2*c, H, W] using adjacent channel pairs."""
    if symbol_tensor.dim() != 5 or symbol_tensor.size(-1) != 2:
        raise ValueError("Expected symbol tensor with shape [B, c, H, W, 2]")

    b, c, h, w, _ = symbol_tensor.shape
    return symbol_tensor.permute(0, 1, 4, 2, 3).contiguous().view(b, 2 * c, h, w)


def flatten_symbol_tensor(symbol_tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """Flatten [B, c, H, W, 2] into [N, 2], returning shape metadata for reconstruction."""
    if symbol_tensor.dim() != 5 or symbol_tensor.size(-1) != 2:
        raise ValueError("Expected symbol tensor with shape [B, c, H, W, 2]")

    b, c, h, w, _ = symbol_tensor.shape
    return symbol_tensor.view(-1, 2), (b, c, h, w)


def unflatten_symbol_tensor(flat_symbol_tensor: torch.Tensor, shape_meta: Tuple[int, int, int, int]) -> torch.Tensor:
    """Rebuild [B, c, H, W, 2] from flattened [N, 2] with metadata."""
    if flat_symbol_tensor.dim() != 2 or flat_symbol_tensor.size(-1) != 2:
        raise ValueError("Expected flattened symbol tensor with shape [N, 2]")

    b, c, h, w = shape_meta
    return flat_symbol_tensor.view(b, c, h, w, 2)


def average_symbol_power(symbol_tensor: torch.Tensor) -> torch.Tensor:
    """Average symbol power E[I^2 + Q^2] for tensors ending with dimension 2."""
    if symbol_tensor.size(-1) != 2:
        raise ValueError("Expected last dimension to be size 2 for I/Q symbol power")
    return symbol_tensor.pow(2).sum(dim=-1).mean()


def normalize_constellation_power(codebook: torch.Tensor, target_power: float = 1.0, eps: float = EPS) -> torch.Tensor:
    """Scale codebook [M, 2] to enforce average symbol power."""
    if codebook.dim() != 2 or codebook.size(-1) != 2:
        raise ValueError("Expected codebook shape [M, 2]")

    current_power = average_symbol_power(codebook)
    scale = torch.sqrt(torch.tensor(target_power, device=codebook.device, dtype=codebook.dtype) / (current_power + eps))
    return codebook * scale


def normalize_symbol_power(symbols: torch.Tensor, target_power: float = 1.0, eps: float = EPS) -> torch.Tensor:
    """Scale symbol tensor (..., 2) to enforce average symbol power."""
    current_power = average_symbol_power(symbols)
    scale = torch.sqrt(torch.tensor(target_power, device=symbols.device, dtype=symbols.dtype) / (current_power + eps))
    return symbols * scale


def compute_codebook_usage_histogram(indices: torch.Tensor, constellation_size: int) -> torch.Tensor:
    return torch.bincount(indices.view(-1), minlength=constellation_size)


def codebook_usage_entropy(usage_counts: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    total = usage_counts.sum().clamp_min(1.0)
    probs = usage_counts.float() / total
    mask = probs > 0
    return -(probs[mask] * torch.log(probs[mask] + eps)).sum()


def nearest_point_distance_stats(
    symbols: torch.Tensor,
    codebook: torch.Tensor,
    nearest_indices: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    if nearest_indices is None:
        distances = torch.sum((symbols[:, None, :] - codebook[None, :, :]) ** 2, dim=-1)
        nearest_indices = torch.argmin(distances, dim=-1)
    nearest_points = codebook.index_select(0, nearest_indices)
    nearest_distance = (symbols - nearest_points).pow(2).sum(dim=-1)
    return {
        "nearest_distance_mean": nearest_distance.mean().item(),
        "nearest_distance_std": nearest_distance.std(unbiased=False).item(),
        "nearest_distance_max": nearest_distance.max().item(),
    }


def map_to_mic_codebook(
    z_tensor: torch.Tensor,
    codebook: torch.Tensor,
    clip_value: float = 2.0,
    power_constraint_mode: str = "none",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standalone hard nearest-neighbor mapping for deployment or SDR integration."""
    z_batched, squeezed = _ensure_batched_latent(z_tensor)
    symbols = pair_channels_to_symbols(z_batched)
    flat_symbols, shape_meta = flatten_symbol_tensor(symbols)

    clipped = flat_symbols.clamp(min=-clip_value, max=clip_value)
    normalized_codebook = normalize_constellation_power(codebook) if power_constraint_mode == "codebook" else codebook

    distances = torch.sum((clipped[:, None, :] - normalized_codebook[None, :, :]) ** 2, dim=-1)
    nearest_indices = torch.argmin(distances, dim=-1)
    mapped_symbols = normalized_codebook.index_select(0, nearest_indices)

    if power_constraint_mode == "post_mapper":
        mapped_symbols = normalize_symbol_power(mapped_symbols)

    mapped = unpair_symbols_to_channels(unflatten_symbol_tensor(mapped_symbols, shape_meta))
    index_shape = (shape_meta[0], shape_meta[1], shape_meta[2], shape_meta[3])
    mapped_indices = nearest_indices.view(*index_shape)

    if squeezed:
        mapped = mapped.squeeze(0)
        mapped_indices = mapped_indices.squeeze(0)

    return mapped, mapped_indices


class MICLayer(nn.Module):
    """Mapping to Irregular Constellation (MIC) with hard/soft surrogate quantization."""

    def __init__(
        self,
        constellation_size: int = 16,
        clip_value: float = 2.0,
        temperature: float = 0.1,
        delta: Optional[float] = None,
        hard_forward: bool = True,
        train_mode: str = "hard_forward_soft_backward",
        power_constraint_mode: str = "codebook",
        init_method: str = "auto",
    ):
        super().__init__()

        if constellation_size <= 1:
            raise ValueError("constellation_size must be > 1")
        if train_mode not in {"soft", "straight_through", "hard_forward_soft_backward"}:
            raise ValueError("Invalid train_mode")
        if power_constraint_mode not in {"codebook", "post_mapper", "none"}:
            raise ValueError("Invalid power_constraint_mode")

        self.constellation_size = int(constellation_size)
        self.clip_value = float(clip_value)
        self.temperature = float(temperature)
        self.delta = None if delta is None else float(delta)
        self.hard_forward = bool(hard_forward)
        self.train_mode = train_mode
        self.power_constraint_mode = power_constraint_mode
        self.deploy_hard = False

        self.codebook = nn.Parameter(self._init_codebook(self.constellation_size, init_method=init_method))
        self.last_stats: Dict[str, object] = {}

    def _init_codebook(self, constellation_size: int, init_method: str = "auto") -> torch.Tensor:
        side = int(round(constellation_size ** 0.5))
        is_square = side * side == constellation_size

        use_qam = init_method in {"qam", "auto"} and is_square
        if use_qam:
            levels = torch.linspace(-(side - 1), side - 1, side)
            grid_i, grid_q = torch.meshgrid(levels, levels, indexing="ij")
            codebook = torch.stack([grid_i.reshape(-1), grid_q.reshape(-1)], dim=-1)
        else:
            codebook = 0.1 * torch.randn(constellation_size, 2)

        return normalize_constellation_power(codebook)

    def get_effective_codebook(self) -> torch.Tensor:
        if self.power_constraint_mode == "codebook":
            return normalize_constellation_power(self.codebook)
        return self.codebook

    def set_deploy_mode(self, enabled: bool = True):
        self.deploy_hard = bool(enabled)

    def set_train_mode(self, train_mode: str):
        if train_mode not in {"soft", "straight_through", "hard_forward_soft_backward"}:
            raise ValueError("Invalid train_mode")
        self.train_mode = train_mode

    def set_temperature(self, temperature: float):
        self.temperature = float(temperature)

    def set_delta(self, delta: Optional[float]):
        self.delta = None if delta is None else float(delta)

    def _distance_matrix(self, symbols: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        return torch.sum((symbols[:, None, :] - codebook[None, :, :]) ** 2, dim=-1)

    def _compute_soft_output(self, distances: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        if self.delta is not None:
            logits = -self.delta * distances
        else:
            logits = -distances / max(self.temperature, EPS)
        weights = F.softmax(logits, dim=-1)
        return weights @ codebook

    def _update_last_stats(
        self,
        clipped_symbols: torch.Tensor,
        mapped_symbols: torch.Tensor,
        codebook: torch.Tensor,
        nearest_indices: torch.Tensor,
        nearest_distances: torch.Tensor,
    ):
        usage_counts = compute_codebook_usage_histogram(nearest_indices, self.constellation_size)
        entropy = codebook_usage_entropy(usage_counts)
        active_fraction = (usage_counts > 0).float().mean()

        min_interpoint = torch.tensor(0.0, device=codebook.device)
        if codebook.size(0) > 1:
            pairwise = torch.cdist(codebook, codebook, p=2)
            mask = ~torch.eye(codebook.size(0), dtype=torch.bool, device=codebook.device)
            min_interpoint = pairwise[mask].min()

        self.last_stats = {
            "usage_counts": usage_counts.detach().cpu(),
            "usage_entropy": entropy.item(),
            "active_fraction": active_fraction.item(),
            "avg_nearest_distance": nearest_distances.mean().item(),
            "mapper_output_power": average_symbol_power(mapped_symbols).item(),
            "codebook_power": average_symbol_power(codebook).item(),
            "min_interpoint_distance": min_interpoint.item(),
            "clip_value": self.clip_value,
            "temperature": self.temperature,
            "delta": self.delta,
            **nearest_point_distance_stats(clipped_symbols, codebook, nearest_indices=nearest_indices),
        }

    def forward(self, z_tensor: torch.Tensor, return_indices: bool = False) -> torch.Tensor:
        z_batched, squeezed = _ensure_batched_latent(z_tensor)

        symbols = pair_channels_to_symbols(z_batched)
        flat_symbols, shape_meta = flatten_symbol_tensor(symbols)
        clipped_symbols = flat_symbols.clamp(min=-self.clip_value, max=self.clip_value)

        effective_codebook = self.get_effective_codebook()
        distances = self._distance_matrix(clipped_symbols, effective_codebook)
        nearest_indices = torch.argmin(distances, dim=-1)
        hard_output = effective_codebook.index_select(0, nearest_indices)
        soft_output = self._compute_soft_output(distances, effective_codebook)

        if self.deploy_hard:
            mapped_symbols = hard_output
        elif self.train_mode == "soft":
            mapped_symbols = soft_output
        elif self.train_mode in {"straight_through", "hard_forward_soft_backward"}:
            if self.hard_forward:
                mapped_symbols = soft_output + (hard_output - soft_output).detach()
            else:
                mapped_symbols = soft_output
        else:
            raise ValueError("Unsupported MIC train mode")

        if self.power_constraint_mode == "post_mapper":
            mapped_symbols = normalize_symbol_power(mapped_symbols)

        nearest_distances = distances.gather(1, nearest_indices.unsqueeze(1)).squeeze(1)
        self._update_last_stats(
            clipped_symbols=clipped_symbols,
            mapped_symbols=mapped_symbols,
            codebook=effective_codebook,
            nearest_indices=nearest_indices,
            nearest_distances=nearest_distances,
        )

        mapped_symbol_tensor = unflatten_symbol_tensor(mapped_symbols, shape_meta)
        mapped = unpair_symbols_to_channels(mapped_symbol_tensor)
        mapped_indices = nearest_indices.view(shape_meta[0], shape_meta[1], shape_meta[2], shape_meta[3])

        if squeezed:
            mapped = mapped.squeeze(0)
            mapped_indices = mapped_indices.squeeze(0)

        if return_indices:
            return mapped, mapped_indices
        return mapped

    @torch.no_grad()
    def export_constellation(self, export_path: str, extra_metadata: Optional[Dict[str, object]] = None) -> Dict[str, str]:
        base = Path(export_path)
        if base.suffix:
            base = base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)

        codebook = self.get_effective_codebook().detach().cpu()
        avg_power = average_symbol_power(codebook).item()

        metadata = {
            "mapper_type": "mic",
            "constellation_size": self.constellation_size,
            "clip_value": self.clip_value,
            "temperature": self.temperature,
            "delta": self.delta,
            "hard_forward": self.hard_forward,
            "train_mode": self.train_mode,
            "deploy_hard": self.deploy_hard,
            "power_constraint_mode": self.power_constraint_mode,
            "average_codebook_power": avg_power,
        }
        if extra_metadata is not None:
            metadata.update(extra_metadata)

        pt_path = str(base) + ".pt"
        npy_path = str(base) + ".npy"
        meta_path = str(base) + ".json"

        torch.save({"codebook": codebook, "metadata": metadata}, pt_path)
        np.save(npy_path, codebook.numpy())
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {
            "pt_path": pt_path,
            "npy_path": npy_path,
            "metadata_path": meta_path,
        }

    def get_stats(self) -> Dict[str, object]:
        return dict(self.last_stats)


class MRCLayer(nn.Module):
    """Mapping to Regular Constellation (MRC) with optional learnable global transform."""

    def __init__(
        self,
        levels_per_axis: int = 4,
        init_bounds: Optional[object] = None,
        clip_value: float = 2.0,
        temperature: float = 0.1,
        delta: Optional[float] = None,
        hard_forward: bool = True,
        train_mode: str = "hard_forward_soft_backward",
        power_constraint_mode: str = "codebook",
        learnable_scale: bool = True,
        learnable_rotation: bool = True,
        learnable_shift: bool = False,
    ):
        super().__init__()

        if levels_per_axis < 2:
            raise ValueError("levels_per_axis must be >= 2")
        if train_mode not in {"soft", "straight_through", "hard_forward_soft_backward"}:
            raise ValueError("Invalid train_mode")
        if power_constraint_mode not in {"codebook", "post_mapper", "none"}:
            raise ValueError("Invalid power_constraint_mode")

        self.levels_per_axis = int(levels_per_axis)
        self.constellation_size = int(self.levels_per_axis ** 2)
        self.init_bounds = _parse_init_bounds(init_bounds)

        self.clip_value = float(clip_value)
        self.temperature = float(temperature)
        self.delta = None if delta is None else float(delta)
        self.hard_forward = bool(hard_forward)
        self.train_mode = train_mode
        self.power_constraint_mode = power_constraint_mode
        self.deploy_hard = False

        base_codebook = _build_regular_codebook(self.levels_per_axis, self.init_bounds)
        self.register_buffer("base_codebook", base_codebook)

        self.learnable_scale = bool(learnable_scale)
        self.learnable_rotation = bool(learnable_rotation)
        self.learnable_shift = bool(learnable_shift)

        if self.learnable_scale:
            self.log_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        else:
            self.register_buffer("log_scale", torch.tensor(0.0, dtype=torch.float32))

        if self.learnable_rotation:
            self.theta = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        else:
            self.register_buffer("theta", torch.tensor(0.0, dtype=torch.float32))

        if self.learnable_shift:
            self.shift = nn.Parameter(torch.zeros(2, dtype=torch.float32))
        else:
            self.register_buffer("shift", torch.zeros(2, dtype=torch.float32))

        self.last_stats: Dict[str, object] = {}

    def _transformed_codebook(self) -> torch.Tensor:
        codebook = self.base_codebook

        scale = torch.exp(self.log_scale)
        codebook = codebook * scale

        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)
        rot = torch.stack([
            torch.stack([cos_t, -sin_t]),
            torch.stack([sin_t, cos_t]),
        ])
        codebook = codebook @ rot.t()

        codebook = codebook + self.shift.view(1, 2)
        return codebook

    def get_effective_codebook(self) -> torch.Tensor:
        codebook = self._transformed_codebook()
        if self.power_constraint_mode == "codebook":
            return normalize_constellation_power(codebook)
        return codebook

    def set_deploy_mode(self, enabled: bool = True):
        self.deploy_hard = bool(enabled)

    def set_train_mode(self, train_mode: str):
        if train_mode not in {"soft", "straight_through", "hard_forward_soft_backward"}:
            raise ValueError("Invalid train_mode")
        self.train_mode = train_mode

    def set_temperature(self, temperature: float):
        self.temperature = float(temperature)

    def set_delta(self, delta: Optional[float]):
        self.delta = None if delta is None else float(delta)

    def _distance_matrix(self, symbols: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        return torch.sum((symbols[:, None, :] - codebook[None, :, :]) ** 2, dim=-1)

    def _compute_soft_output(self, distances: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        if self.delta is not None:
            logits = -self.delta * distances
        else:
            logits = -distances / max(self.temperature, EPS)
        weights = F.softmax(logits, dim=-1)
        return weights @ codebook

    def _update_last_stats(
        self,
        clipped_symbols: torch.Tensor,
        mapped_symbols: torch.Tensor,
        codebook: torch.Tensor,
        nearest_indices: torch.Tensor,
        nearest_distances: torch.Tensor,
    ):
        usage_counts = compute_codebook_usage_histogram(nearest_indices, self.constellation_size)
        entropy = codebook_usage_entropy(usage_counts)
        active_fraction = (usage_counts > 0).float().mean()

        min_interpoint = torch.tensor(0.0, device=codebook.device)
        if codebook.size(0) > 1:
            pairwise = torch.cdist(codebook, codebook, p=2)
            mask = ~torch.eye(codebook.size(0), dtype=torch.bool, device=codebook.device)
            min_interpoint = pairwise[mask].min()

        self.last_stats = {
            "usage_counts": usage_counts.detach().cpu(),
            "usage_entropy": entropy.item(),
            "active_fraction": active_fraction.item(),
            "avg_nearest_distance": nearest_distances.mean().item(),
            "mapper_output_power": average_symbol_power(mapped_symbols).item(),
            "codebook_power": average_symbol_power(codebook).item(),
            "min_interpoint_distance": min_interpoint.item(),
            "clip_value": self.clip_value,
            "temperature": self.temperature,
            "delta": self.delta,
            **nearest_point_distance_stats(clipped_symbols, codebook, nearest_indices=nearest_indices),
        }

    def forward(self, z_tensor: torch.Tensor, return_indices: bool = False) -> torch.Tensor:
        z_batched, squeezed = _ensure_batched_latent(z_tensor)

        symbols = pair_channels_to_symbols(z_batched)
        flat_symbols, shape_meta = flatten_symbol_tensor(symbols)
        clipped_symbols = flat_symbols.clamp(min=-self.clip_value, max=self.clip_value)

        effective_codebook = self.get_effective_codebook()
        distances = self._distance_matrix(clipped_symbols, effective_codebook)
        nearest_indices = torch.argmin(distances, dim=-1)
        hard_output = effective_codebook.index_select(0, nearest_indices)
        soft_output = self._compute_soft_output(distances, effective_codebook)

        if self.deploy_hard:
            mapped_symbols = hard_output
        elif self.train_mode == "soft":
            mapped_symbols = soft_output
        elif self.train_mode in {"straight_through", "hard_forward_soft_backward"}:
            if self.hard_forward:
                mapped_symbols = soft_output + (hard_output - soft_output).detach()
            else:
                mapped_symbols = soft_output
        else:
            raise ValueError("Unsupported MRC train mode")

        if self.power_constraint_mode == "post_mapper":
            mapped_symbols = normalize_symbol_power(mapped_symbols)

        nearest_distances = distances.gather(1, nearest_indices.unsqueeze(1)).squeeze(1)
        self._update_last_stats(
            clipped_symbols=clipped_symbols,
            mapped_symbols=mapped_symbols,
            codebook=effective_codebook,
            nearest_indices=nearest_indices,
            nearest_distances=nearest_distances,
        )

        mapped_symbol_tensor = unflatten_symbol_tensor(mapped_symbols, shape_meta)
        mapped = unpair_symbols_to_channels(mapped_symbol_tensor)
        mapped_indices = nearest_indices.view(shape_meta[0], shape_meta[1], shape_meta[2], shape_meta[3])

        if squeezed:
            mapped = mapped.squeeze(0)
            mapped_indices = mapped_indices.squeeze(0)

        if return_indices:
            return mapped, mapped_indices
        return mapped

    @torch.no_grad()
    def export_constellation(self, export_path: str, extra_metadata: Optional[Dict[str, object]] = None) -> Dict[str, str]:
        base = Path(export_path)
        if base.suffix:
            base = base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)

        codebook = self.get_effective_codebook().detach().cpu()
        avg_power = average_symbol_power(codebook).item()

        metadata = {
            "mapper_type": "mrc",
            "levels_per_axis": self.levels_per_axis,
            "constellation_size": self.constellation_size,
            "init_bounds": [self.init_bounds[0], self.init_bounds[1]],
            "clip_value": self.clip_value,
            "temperature": self.temperature,
            "delta": self.delta,
            "hard_forward": self.hard_forward,
            "train_mode": self.train_mode,
            "deploy_hard": self.deploy_hard,
            "power_constraint_mode": self.power_constraint_mode,
            "learnable_scale": self.learnable_scale,
            "learnable_rotation": self.learnable_rotation,
            "learnable_shift": self.learnable_shift,
            "average_codebook_power": avg_power,
        }
        if extra_metadata is not None:
            metadata.update(extra_metadata)

        pt_path = str(base) + ".pt"
        npy_path = str(base) + ".npy"
        meta_path = str(base) + ".json"

        torch.save({"codebook": codebook, "metadata": metadata}, pt_path)
        np.save(npy_path, codebook.numpy())
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return {
            "pt_path": pt_path,
            "npy_path": npy_path,
            "metadata_path": meta_path,
        }

    def get_stats(self) -> Dict[str, object]:
        return dict(self.last_stats)
