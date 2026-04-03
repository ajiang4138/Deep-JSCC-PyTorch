#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Constellation plot visualizer for Deep JSCC encoder output

This script loads a checkpoint and creates a constellation plot
of the signal after the encoder step (before channel transmission).
"""

import argparse
import os
from collections import OrderedDict
from pathlib import Path

# Handle matplotlib before numpy
import matplotlib
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

matplotlib.use('Agg')  # Use non-interactive backend
# Add the project root to path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from constellation import pair_channels_to_symbols
from model import DeepJSCC
from utils import image_normalization


class EncoderHook:
    """Hook to capture encoder output"""
    def __init__(self):
        self.features = None
    
    def __call__(self, module, input, output):
        self.features = output.detach()


def load_checkpoint(checkpoint_path, device='cpu'):
    """Load model from checkpoint, handling DataParallel wrapper"""
    checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint_obj, dict) and 'model_state_dict' in checkpoint_obj:
        state_dict = checkpoint_obj['model_state_dict']
    else:
        state_dict = checkpoint_obj
    
    # Remove 'module.' prefix if model was saved with DataParallel
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    
    return new_state_dict


def get_checkpoint_params(checkpoint_dir):
    """Extract parameters from checkpoint directory name"""
    # Format: CIFAR10_c_snr_ratio_channel_timestamp
    parts = Path(checkpoint_dir).name.split('_')
    try:
        c = int(parts[1])
        snr = float(parts[2])
        ratio = float(parts[3])
        channel = parts[4]
        return c, snr, ratio, channel
    except (IndexError, ValueError):
        raise ValueError(f"Could not parse checkpoint directory: {checkpoint_dir}")


def find_awgn_checkpoints(checkpoint_dir):
    """Find all AWGN checkpoints in directory"""
    checkpoint_dir = Path(checkpoint_dir)
    awgn_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and 'AWGN' in d.name]
    return sorted(awgn_dirs)


def create_constellation_plot(encoder_output, title="Encoder Output Constellation", save_path=None):
    """
    Create an overlay constellation plot showing all complex symbols together.
    
    This function visualizes the I-Q distribution of all encoded symbols on a single plot.
    Each symbol is represented by a different color, showing the learned transmission 
    strategy of the encoder.
    
    Args:
        encoder_output (torch.Tensor): Encoder output with shape:
            - (batch, 2*c, height, width): 4D tensor from batch processing
            - (2*c, height, width): 3D tensor from single sample
            - (num_symbols, 2*c): 2D tensor of pre-reshaped symbols
        title (str): Title for the constellation plot
        save_path (str, optional): Path to save the figure. If None, figure is not saved.
    
    Returns:
        tuple: (fig, ax) - matplotlib figure and axis objects
    """
    # Reshape encoder output to (num_symbols, 2*c) format
    # where num_symbols is the total number of transmitted symbols
    # and 2*c are the I and Q components for c complex symbols
    if encoder_output.dim() == 4:
        # Input: (batch, channels, height, width)
        batch, channels, height, width = encoder_output.shape
        c = channels // 2
        paired = pair_channels_to_symbols(encoder_output)
        symbols = paired.permute(0, 2, 3, 1, 4).reshape(batch * height * width, c, 2)
    elif encoder_output.dim() == 3:
        # Input: (channels, height, width)
        channels, height, width = encoder_output.shape
        c = channels // 2
        paired = pair_channels_to_symbols(encoder_output)
        symbols = paired.permute(1, 2, 0, 3).reshape(height * width, c, 2)
    elif encoder_output.dim() == 2:
        # Input: (num_symbols, channels)
        symbols = encoder_output
        num_symbols, channels = symbols.shape
        c = channels // 2
        symbols = symbols.view(num_symbols, c, 2)
    else:
        raise ValueError(f"Unexpected encoder output dimension: {encoder_output.dim()}")
    
    # Create a single large figure for better visibility
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Define colors for each symbol for clear visual distinction
    colors = plt.cm.Set1(np.linspace(0, 1, c))
    
    # Plot each complex symbol with different color
    for i in range(c):
        # Adjacent channels are treated as one complex symbol pair (I, Q)
        I = symbols[:, i, 0].cpu().numpy()  # In-phase component
        Q = symbols[:, i, 1].cpu().numpy()  # Quadrature component
        
        # Create scatter plot for this symbol
        ax.scatter(I, Q, alpha=0.6, s=20, label=f'Symbol {i+1}', color=colors[i], edgecolors='none')
    
    # Configure plot appearance
    ax.set_xlabel('I (In-phase)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Q (Quadrature)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add reference axes at origin
    ax.axhline(y=0, color='k', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='k', linewidth=0.8, alpha=0.5)
    
    # Set equal aspect ratio so circles appear as circles
    ax.set_aspect('equal', adjustable='box')
    
    # Add legend for symbol identification
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Constellation plot saved to: {save_path}")
    
    return fig, ax


def visualize_checkpoint(checkpoint_path, test_image_path=None, snr=2000, device='cpu'):
    """
    Visualize constellation from a checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint .pkl file
        test_image_path: Path to test image (if None, uses random tensor)
        snr: SNR value for channel
        device: Device to use (cuda or cpu)
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    # Extract parameters from checkpoint directory naming convention
    # Format: CIFAR10_c_snr_ratio_channel_timestamp
    checkpoint_dir = Path(checkpoint_path).parent
    c, snr_ckpt, ratio, channel = get_checkpoint_params(checkpoint_dir)
    print(f"Checkpoint params - c: {c}, SNR: {snr_ckpt}, Ratio: {ratio}, Channel: {channel}")
    
    # ============================================================================
    # Load and initialize model
    # ============================================================================
    # Create DeepJSCC model with extracted parameters
    model = DeepJSCC(c=c, channel_type=channel, snr=snr)
    # Load trained weights from checkpoint
    state_dict = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    # Set to evaluation mode (disables dropout, batch norm updates, etc.)
    model.eval()
    
    # ============================================================================
    # Prepare input: load test image or generate random tensor
    # ============================================================================
    if test_image_path and Path(test_image_path).exists():
        print(f"Loading test image: {test_image_path}")
        image = Image.open(test_image_path)
        transform = transforms.Compose([transforms.ToTensor()])
        x = transform(image).to(device)
        # Add batch dimension: (C, H, W) -> (1, C, H, W)
        x = x.unsqueeze(0)
    else:
        print("Using random input tensor")
        # Generate random input tensor with shape (batch=1, channels=3, height=256, width=256)
        x = torch.randn(1, 3, 256, 256).to(device)
    
    # ============================================================================
    # Capture encoder output using hook
    # ============================================================================
    # Register hook to capture encoder's intermediate output
    hook = EncoderHook()
    model.encoder.register_forward_hook(hook)
    
    # ============================================================================
    # Forward pass: compute encoder output WITHOUT channel transmission
    # ============================================================================
    with torch.no_grad():
        # This gives us the normalized symbols before noise is added
        encoder_output = model.encoder(x)
    
    print(f"Encoder output shape: {encoder_output.shape}")
    
    # ============================================================================
    # Generate and save constellation plot
    # ============================================================================
    # Create filename following naming convention: constellation_c{c}_snr{snr}dB_ratio{ratio}_{channel}.png
    constellation_dir = Path('./constellations')
    constellation_dir.mkdir(exist_ok=True)
    
    filename = f"constellation_c{c}_snr{snr_ckpt:.1f}dB_ratio{ratio:.2f}_{channel}.png"
    save_path = str(constellation_dir / filename)
    
    # Create descriptive title with all key parameters
    title = f"Encoder Constellation - c={c}, SNR={snr_ckpt}dB, Ratio={ratio}, Channel={channel}"
    
    fig, ax = create_constellation_plot(encoder_output, title=title, save_path=save_path)
    print(f"Constellation plot saved successfully")
    
    return encoder_output


def batch_visualize_awgn_checkpoints(checkpoint_base_dir, test_image_path=None, device='cpu', 
                                    max_checkpoints=None):
    """
    Visualize constellation for multiple AWGN checkpoints
    
    Args:
        checkpoint_base_dir: Base directory containing checkpoints
        test_image_path: Path to test image
        device: Device to use
        max_checkpoints: Maximum number of checkpoints to process
    """
    checkpoint_dirs = find_awgn_checkpoints(checkpoint_base_dir)
    print(f"Found {len(checkpoint_dirs)} AWGN checkpoints")
    
    # Limit number of checkpoints if specified
    if max_checkpoints:
        checkpoint_dirs = checkpoint_dirs[:max_checkpoints]
    
    # Process each checkpoint
    for ckpt_dir in checkpoint_dirs:
        try:
            # Find all epoch files in this checkpoint directory
            # Epoch files follow naming pattern: epoch_*.pkl
            epoch_files = list(ckpt_dir.glob('epoch_*.pkl'))
            if not epoch_files:
                print(f"No checkpoint files found in {ckpt_dir}")
                continue
            
            # Use the latest (last) epoch for visualization
            epoch_file = sorted(epoch_files)[-1]
            print(f"\nProcessing: {ckpt_dir.name}")
            visualize_checkpoint(str(epoch_file), test_image_path, device=device)
            
        except Exception as e:
            print(f"Error processing {ckpt_dir.name}: {e}")
            continue


def main():
    parser = argparse.ArgumentParser(description='Deep JSCC Encoder Constellation Visualizer')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file (epoch_*.pkl)')
    parser.add_argument('--checkpoint-dir', type=str, 
                        default='./out/checkpoints',
                        help='Base checkpoint directory (for batch processing)')
    parser.add_argument('--image', type=str, 
                        default='./demo/kodim08.png',
                        help='Path to test image')
    parser.add_argument('--snr', type=float, default=2000, help='SNR for evaluation')
    parser.add_argument('--batch', action='store_true', help='Process all AWGN checkpoints')
    parser.add_argument('--max-checkpoints', type=int, default=None,
                        help='Maximum number of checkpoints to process (for batch mode)')
    parser.add_argument('--device', type=str, default='cpu', 
                        choices=['cpu', 'cuda'], help='Device to use')
    
    args = parser.parse_args()
    
    # ============================================================================
    # Validate and setup computation device
    # ============================================================================
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # ============================================================================
    # Execution modes
    # ============================================================================
    if args.batch:
        # Batch process all AWGN checkpoints
        batch_visualize_awgn_checkpoints(args.checkpoint_dir, args.image, 
                                        device=args.device, 
                                        max_checkpoints=args.max_checkpoints)
    elif args.checkpoint:
        # Process single checkpoint
        visualize_checkpoint(args.checkpoint, args.image, args.snr, device=args.device)
    else:
        # If no specific checkpoint, try to find the latest AWGN checkpoint
        checkpoint_dirs = find_awgn_checkpoints(args.checkpoint_dir)
        if checkpoint_dirs:
            latest_dir = checkpoint_dirs[-1]
            epoch_files = list(latest_dir.glob('epoch_*.pkl'))
            if epoch_files:
                latest_checkpoint = sorted(epoch_files)[-1]
                print(f"Using latest AWGN checkpoint: {latest_checkpoint}")
                visualize_checkpoint(str(latest_checkpoint), args.image, args.snr, device=args.device)
            else:
                print(f"No checkpoint files found in {latest_dir}")
        else:
            print(f"No AWGN checkpoints found in {args.checkpoint_dir}")
            print("\nUsage examples:")
            print("  Single checkpoint:   python signal_constellation_visualizer.py --checkpoint ./out/checkpoints/CIFAR10_4_1.0_0.08_AWGN_22h13m53s_on_Jun_07_2024/epoch_999.pkl")
            print("  Batch all AWGN:      python signal_constellation_visualizer.py --batch")
            print("  Batch with image:    python signal_constellation_visualizer.py --batch --image ./demo/kodim08.png")


if __name__ == '__main__':
    main()
