import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset import Vanilla
from model import DeepJSCC
from train import evaluate_epoch, str2bool
from utils import get_psnr


def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and 'model_state_dict' in checkpoint_obj:
        return checkpoint_obj['model_state_dict'], checkpoint_obj
    return checkpoint_obj, {}


def strip_module_prefix(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}


def load_checkpoint_flexible(model, checkpoint_path, device='cpu', strict=False):
    checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    state_dict, metadata = extract_state_dict(checkpoint_obj)
    load_result = model.load_state_dict(strip_module_prefix(state_dict), strict=strict)
    return load_result, metadata


def build_test_loader(dataset_name, params):
    if dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.CIFAR10(root='../dataset/', train=False, download=True, transform=transform)
    elif dataset_name == 'imagenet':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((128, 128))])
        test_dataset = Vanilla(root='../dataset/ImageNet/val', transform=transform)
    else:
        raise Exception('Unknown dataset')

    return DataLoader(
        test_dataset,
        shuffle=True,
        batch_size=params['batch_size'],
        num_workers=params['num_workers'],
    )


def mapper_kwargs_from_dict(mapper_cfg, args):
    if mapper_cfg.get('mapper_type', 'none') == 'none':
        return None
    return {
        'constellation_size': mapper_cfg.get('constellation_size', args.constellation_size),
        'clip_value': mapper_cfg.get('clip_value', args.mapper_clip_value),
        'temperature': mapper_cfg.get('temperature', args.mic_temperature),
        'delta': mapper_cfg.get('delta', args.mic_delta),
        'hard_forward': mapper_cfg.get('hard_forward', args.mic_hard_forward),
        'train_mode': mapper_cfg.get('train_mode', args.mic_train_mode),
        'power_constraint_mode': mapper_cfg.get('power_constraint_mode', args.power_constraint_mode),
    }


def resolve_mapper_config(config_params, checkpoint_meta, args):
    mapper_cfg = {
        'mapper_type': config_params.get('mapper_type', 'none'),
        'constellation_size': config_params.get('constellation_size', args.constellation_size),
        'clip_value': config_params.get('mapper_clip_value', args.mapper_clip_value),
        'temperature': config_params.get('mic_temperature', args.mic_temperature),
        'delta': config_params.get('mic_delta', args.mic_delta),
        'hard_forward': config_params.get('mic_hard_forward', args.mic_hard_forward),
        'train_mode': config_params.get('mic_train_mode', args.mic_train_mode),
        'power_constraint_mode': config_params.get('power_constraint_mode', args.power_constraint_mode),
    }

    if checkpoint_meta and isinstance(checkpoint_meta, dict):
        ckpt_params = checkpoint_meta.get('params', {})
        mapper_cfg.update({
            'mapper_type': ckpt_params.get('mapper_type', mapper_cfg['mapper_type']),
            'constellation_size': ckpt_params.get('constellation_size', mapper_cfg['constellation_size']),
            'clip_value': ckpt_params.get('mapper_clip_value', mapper_cfg['clip_value']),
            'temperature': ckpt_params.get('mic_temperature', mapper_cfg['temperature']),
            'delta': ckpt_params.get('mic_delta', mapper_cfg['delta']),
            'hard_forward': ckpt_params.get('mic_hard_forward', mapper_cfg['hard_forward']),
            'train_mode': ckpt_params.get('mic_train_mode', mapper_cfg['train_mode']),
            'power_constraint_mode': ckpt_params.get('power_constraint_mode', mapper_cfg['power_constraint_mode']),
        })

        mapper_ckpt = checkpoint_meta.get('mapper_config', {})
        if isinstance(mapper_ckpt, dict):
            mapper_cfg.update({
                'mapper_type': mapper_ckpt.get('mapper_type', mapper_cfg['mapper_type']),
                'constellation_size': mapper_ckpt.get('constellation_size', mapper_cfg['constellation_size']),
                'clip_value': mapper_ckpt.get('clip_value', mapper_cfg['clip_value']),
                'temperature': mapper_ckpt.get('temperature', mapper_cfg['temperature']),
                'delta': mapper_ckpt.get('delta', mapper_cfg['delta']),
                'hard_forward': mapper_ckpt.get('hard_forward', mapper_cfg['hard_forward']),
                'train_mode': mapper_ckpt.get('train_mode', mapper_cfg['train_mode']),
                'power_constraint_mode': mapper_ckpt.get('power_constraint_mode', mapper_cfg['power_constraint_mode']),
            })

    if args.mapper_type != 'none':
        mapper_cfg['mapper_type'] = args.mapper_type
        mapper_cfg['constellation_size'] = args.constellation_size
        mapper_cfg['clip_value'] = args.mapper_clip_value
        mapper_cfg['temperature'] = args.mic_temperature
        mapper_cfg['delta'] = args.mic_delta
        mapper_cfg['hard_forward'] = args.mic_hard_forward
        mapper_cfg['train_mode'] = args.mic_train_mode
        mapper_cfg['power_constraint_mode'] = args.power_constraint_mode

    return mapper_cfg


def eval_snr(model, test_loader, writer, param, eval_mode='train_surrogate', times=10):
    snr_list = range(0, 26, 1)
    for snr in snr_list:
        model.change_channel(param['channel'], snr)
        model.set_mapper_deploy_mode(eval_mode == 'hard_deploy')

        test_loss = 0.0
        for _ in range(times):
            test_loss += evaluate_epoch(model, param, test_loader)

        test_loss /= times
        psnr = get_psnr(image=None, gt=None, mse=test_loss)
        writer.add_scalar('psnr', psnr, snr)


@torch.no_grad()
def export_mapper_eval_artifacts(model, data_loader, params, args, run_name):
    if model.mapper is None:
        return

    images, _ = next(iter(data_loader))
    images = images.to(params['device'])
    z = model.encoder(images)
    mapped, indices = model.mapper(z, return_indices=True)

    if args.export_quantized_latent_path:
        base = Path(args.export_quantized_latent_path)
        base.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(base.with_name(base.stem + '_' + run_name + '.npy')), mapped.detach().cpu().numpy())

    if args.export_occupancy_path:
        counts = torch.bincount(indices.view(-1), minlength=model.mapper.constellation_size).cpu().numpy().tolist()
        occupancy = {
            'run_name': run_name,
            'counts': counts,
            'active_fraction': float((np.array(counts) > 0).mean()),
            'eval_mode': args.eval_mode,
        }
        occupancy_path = Path(args.export_occupancy_path)
        occupancy_path.parent.mkdir(parents=True, exist_ok=True)
        if occupancy_path.suffix:
            occupancy_path = occupancy_path.with_name(occupancy_path.stem + '_' + run_name + occupancy_path.suffix)
        else:
            occupancy_path = occupancy_path.with_name(occupancy_path.name + '_' + run_name + '.json')
        with open(str(occupancy_path), 'w', encoding='utf-8') as f:
            json.dump(occupancy, f, indent=2)

    if args.export_constellation_path:
        export_base = str(Path(args.export_constellation_path).with_suffix('')) + '_' + run_name
        model.export_mapper_state(export_base, extra_metadata={
            'eval_mode': args.eval_mode,
            'run_name': run_name,
        })


def process_config(config_path, output_dir, args):
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
        params = config['params']
        c = config['inner_channel']
        dataset_name = config['dataset_name']

    params['device'] = args.device
    params['channel'] = args.channel

    test_loader = build_test_loader(dataset_name, params)
    run_name = os.path.splitext(os.path.basename(config_path))[0]
    writer = SummaryWriter(os.path.join(output_dir, 'eval', run_name))

    pkl_list = sorted(glob.glob(os.path.join(output_dir, 'checkpoints', run_name, '*.pkl')))
    if len(pkl_list) == 0:
        raise FileNotFoundError('No checkpoints found for {}'.format(run_name))
    checkpoint_path = pkl_list[-1]

    checkpoint_obj = torch.load(checkpoint_path, map_location=args.device)
    _, checkpoint_meta = extract_state_dict(checkpoint_obj)
    mapper_cfg = resolve_mapper_config(params, checkpoint_meta, args)

    model = DeepJSCC(
        c=c,
        channel_type=params['channel'],
        snr=params.get('snr', None),
        mapper_type=mapper_cfg['mapper_type'],
        mapper_kwargs=mapper_kwargs_from_dict(mapper_cfg, args),
    ).to(params['device'])

    load_result, _ = load_checkpoint_flexible(model, checkpoint_path, device=args.device, strict=False)
    print('Evaluating checkpoint: {}'.format(checkpoint_path))
    print('Missing keys: {}'.format(load_result.missing_keys))
    print('Unexpected keys: {}'.format(load_result.unexpected_keys))

    if model.mapper is not None:
        model.mapper.set_train_mode(mapper_cfg['train_mode'])
        model.mapper.hard_forward = mapper_cfg['hard_forward']
        model.mapper.set_temperature(mapper_cfg['temperature'])
        model.mapper.set_delta(mapper_cfg['delta'])

    eval_snr(model, test_loader, writer, params, eval_mode=args.eval_mode, times=args.times)
    export_mapper_eval_artifacts(model, test_loader, params, args, run_name)
    writer.close()


def parse_c_from_checkpoint_dir(checkpoint_path):
    parts = Path(checkpoint_path).parent.name.split('_')
    if len(parts) < 2:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def evaluate_single_checkpoint(args):
    checkpoint_obj = torch.load(args.checkpoint_path, map_location=args.device)
    _, checkpoint_meta = extract_state_dict(checkpoint_obj)

    c = args.inner_channel
    if c is None and isinstance(checkpoint_meta, dict):
        c = checkpoint_meta.get('inner_channel', None)
    if c is None:
        c = parse_c_from_checkpoint_dir(args.checkpoint_path)
    if c is None:
        raise ValueError('Cannot infer inner channel c. Please provide --inner_channel.')

    params = {
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'device': args.device,
        'channel': args.channel,
        'parallel': False,
    }

    test_loader = build_test_loader(args.dataset, params)
    mapper_cfg = resolve_mapper_config({}, checkpoint_meta, args)

    model = DeepJSCC(
        c=c,
        channel_type=args.channel,
        snr=None,
        mapper_type=mapper_cfg['mapper_type'],
        mapper_kwargs=mapper_kwargs_from_dict(mapper_cfg, args),
    ).to(args.device)

    load_result, _ = load_checkpoint_flexible(model, args.checkpoint_path, device=args.device, strict=False)
    print('Evaluating checkpoint: {}'.format(args.checkpoint_path))
    print('Missing keys: {}'.format(load_result.missing_keys))
    print('Unexpected keys: {}'.format(load_result.unexpected_keys))

    if model.mapper is not None:
        model.mapper.set_train_mode(mapper_cfg['train_mode'])
        model.mapper.hard_forward = mapper_cfg['hard_forward']
        model.mapper.set_temperature(mapper_cfg['temperature'])
        model.mapper.set_delta(mapper_cfg['delta'])

    run_name = Path(args.checkpoint_path).stem
    writer = SummaryWriter(os.path.join(args.output_dir, 'eval', run_name))
    eval_snr(model, test_loader, writer, params, eval_mode=args.eval_mode, times=args.times)
    export_mapper_eval_artifacts(model, test_loader, params, args, run_name)
    writer.close()


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./out', type=str)
    parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'imagenet'])
    parser.add_argument('--channel', default='AWGN', type=str, choices=['AWGN', 'Rayleigh'])
    parser.add_argument('--times', default=10, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--config_path', default='', type=str,
                        help='optional single config yaml path')
    parser.add_argument('--checkpoint_path', default='', type=str,
                        help='optional single checkpoint path')
    parser.add_argument('--inner_channel', default=None, type=int,
                        help='required for direct checkpoint eval if c cannot be parsed from name')

    parser.add_argument('--mapper_type', default='none', type=str, choices=['none', 'mic'])
    parser.add_argument('--constellation_size', default=16, type=int)
    parser.add_argument('--mic_temperature', default=0.1, type=float)
    parser.add_argument('--mic_delta', default=None, type=float)
    parser.add_argument('--mic_hard_forward', default=True, type=str2bool)
    parser.add_argument('--mic_train_mode', default='hard_forward_soft_backward', type=str,
                        choices=['soft', 'straight_through', 'hard_forward_soft_backward'])
    parser.add_argument('--mapper_clip_value', default=2.0, type=float)
    parser.add_argument('--mapper_finetune_from', default='', type=str)
    parser.add_argument('--freeze_encoder', default=False, type=str2bool)
    parser.add_argument('--freeze_decoder', default=False, type=str2bool)
    parser.add_argument('--freeze_mapper', default=False, type=str2bool)
    parser.add_argument('--mapper_lr', default=None, type=float)
    parser.add_argument('--power_constraint_mode', default='codebook', type=str,
                        choices=['codebook', 'post_mapper', 'none'])
    parser.add_argument('--export_constellation_path', default='', type=str)
    parser.add_argument('--mrc_levels_per_axis', default=4, type=int)
    parser.add_argument('--mrc_init_bounds', default='', type=str)

    parser.add_argument('--eval_mode', default='train_surrogate', type=str,
                        choices=['train_surrogate', 'hard_deploy'])
    parser.add_argument('--export_quantized_latent_path', default='', type=str)
    parser.add_argument('--export_occupancy_path', default='', type=str)
    return parser.parse_args()


def main():
    args = config_parser()

    if args.checkpoint_path:
        evaluate_single_checkpoint(args)
        return

    if args.config_path:
        process_config(args.config_path, args.output_dir, args)
        return

    config_dir = os.path.join(args.output_dir, 'configs')
    if not os.path.exists(config_dir):
        raise FileNotFoundError('Config directory not found: {}'.format(config_dir))

    config_files = [
        os.path.join(config_dir, name)
        for name in os.listdir(config_dir)
        if (args.dataset in name or args.dataset.upper() in name)
        and args.channel in name
        and name.endswith('.yaml')
    ]
    config_files = sorted(config_files)

    for config_path in config_files:
        process_config(config_path, args.output_dir, args)


if __name__ == '__main__':
    main()
