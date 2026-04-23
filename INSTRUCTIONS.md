# Instructions for Using MIC and MRC Layers

This guide explains how to use the MIC (Mapping to Irregular Constellation) and MRC (Mapping to Regular Constellation) layers in this repository. These layers are used for advanced symbol mapping in the DeepJSCC framework.

## 1. Overview
- **MICLayer**: Implements mapping to an irregular, learnable constellation. Useful for tasks where the constellation points are optimized during training.
- **MRCLayer**: Implements mapping to a regular (e.g., QAM-like) constellation, with optional learnable global transformations (scale, rotation, shift).

Both layers are implemented in `constellation.py` and can be selected via the `mapper_type` argument in the main model (`DeepJSCC`).

## 2. How to Use

### Selecting a Mapper in DeepJSCC

When initializing the model, choose the mapper type:

```python
from model import DeepJSCC

# For MIC
model = DeepJSCC(c=4, channel_type='AWGN', snr=10, mapper_type='mic')

# For MRC
model = DeepJSCC(c=4, channel_type='AWGN', snr=10, mapper_type='mrc')
```

You can also pass additional keyword arguments for fine-tuning the mapper:

```python
model = DeepJSCC(c=4, channel_type='AWGN', snr=10, mapper_type='mic', mapper_kwargs={
    'constellation_size': 16,
    'clip_value': 2.0,
    'temperature': 0.1,
    'power_constraint_mode': 'codebook',
})
```

### Forward Pass

The mapper is automatically applied in the model's forward pass:

```python
x_hat = model(x)
```

For debugging and visualization, use:

```python
out = model.forward_debug(x, return_mapper_indices=True)
# out['z_pre_mapper'], out['z_post_mapper'], out['mapper_indices']
```

## 3. Visualization and Testing

- Use `mic_image_triplet_test.py` and `mrc_image_triplet_test.py` for visualizing and comparing the effect of MIC and MRC layers.
- Use `mic_mapper_visualizer.py` for detailed debugging and visualization of the mapping process.


## 4. Command-Line Usage (CLI)

This repository provides several scripts for training, evaluation, and visualization. Below are the main scripts and their key arguments:

### Training with MIC/MRC (`train.py`)

Example:
```bash
python train.py --dataset cifar10 --channel AWGN --mapper_type mic --constellation_size 16
```

Key arguments:
- `--dataset`: Dataset to use (`cifar10`, `imagenet`)
- `--channel`: Channel type (`AWGN`, `Rayleigh`)
- `--snr_list`: List of SNR values (e.g., `--snr_list 19 13 7 4 1`)
- `--ratio_list`: List of compression ratios (e.g., `--ratio_list 1/6 1/12`)
- `--mapper_type`: Type of mapper (`none`, `mic`, `mrc`)
- `--constellation_size`: Number of constellation points (MIC)
- `--mic_temperature`, `--mic_delta`, `--mic_train_mode`, `--mapper_clip_value`, `--power_constraint_mode`, etc.: Advanced MIC/MRC options

Run `python train.py --help` for a full list of options.

### MIC Triplet Visualization (`mic_image_triplet_test.py`)

Example:
```bash
python mic_image_triplet_test.py --checkpoint path/to/model.pth --images path/to/img1.png path/to/img2.png --output_dir ./out/test_mic_triplet
```

Key arguments:
- `--checkpoint`: Path to model checkpoint
- `--images`: One or more input image paths
- `--output_dir`: Directory to save outputs
- `--device`: `cpu` or `cuda`
- `--prefer_trained_mapper`: Use trained mapper if available
- `--mic_constellation_size`: Size for estimated codebook
- `--mic_clip_value`: Clipping value for MIC
- `--mic_power_constraint_mode`: Power normalization (`none`, `codebook`, `post_mapper`)

### MRC Triplet Visualization (`mrc_image_triplet_test.py`)

Example:
```bash
python mrc_image_triplet_test.py --checkpoint path/to/model.pth --images path/to/img1.png path/to/img2.png --output_dir ./out-mrc/test_mrc_triplet
```

Key arguments:
- `--checkpoint`: Path to model checkpoint
- `--images`: One or more input image paths
- `--output_dir`: Directory to save outputs
- `--device`: `cpu` or `cuda`
- `--mrc_levels_per_axis`: Levels per I/Q axis (constellation size = levels^2)
- `--mrc_init_bounds`: Initial bounds for MRC grid (e.g., `-1.5,1.5`)
- `--mrc_power_constraint_mode`: Power normalization (`none`, `codebook`, `post_mapper`)

### MIC Mapper Visualizer (`mic_mapper_visualizer.py`)

Example:
```bash
python mic_mapper_visualizer.py --checkpoint path/to/model.pth --input_image path/to/img.png --output_dir ./out/debug_mic_visualizer
```

Key arguments:
- `--checkpoint`: Path to model checkpoint
- `--input_image`: Input image path
- `--output_dir`: Directory for outputs
- `--device`: `cpu` or `cuda`
- `--inner_channel`, `--channel`, `--snr`: Optional overrides
- `--resize_h`, `--resize_w`: Resize input image
- `--max_points`: Max I/Q points in plots
- `--overlay_k`: Overlay decision centers if codebook unavailable

Run any script with `--help` to see all available options and their descriptions.

## 5. File Responsibilities
- `constellation.py`: Core implementation of MICLayer and MRCLayer.
- `model.py`: Main DeepJSCC model, integrates the mapper layers.
- `mic_image_triplet_test.py`, `mrc_image_triplet_test.py`: Scripts for visualizing and comparing MIC/MRC.
- `mic_mapper_visualizer.py`: Utility for visualizing codebooks and symbol mappings.

## 6. References
- See the docstrings in `constellation.py` for detailed parameter explanations.
- For more examples, see the test scripts and visualizer utilities.
