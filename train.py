# -*- coding: utf-8 -*-
"""
Created on Tue Dec  17:00:00 2023

@author: chun
"""
import glob
import json
import os
import time
from fractions import Fraction

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.parallel import DataParallel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from dataset import Vanilla
from model import DeepJSCC, ratio2filtersize
from utils import image_normalization, save_model, set_seed, view_model_param


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in {'true', '1', 'yes', 'y', 't'}:
        return True
    if v in {'false', '0', 'no', 'n', 'f'}:
        return False
    raise ValueError('Boolean value expected.')


def unwrap_model(model):
    return model.module if isinstance(model, DataParallel) else model


def get_mapper_kwargs_from_params(params):
    if params.get('mapper_type', 'none') == 'none':
        return None
    return {
        'constellation_size': params['constellation_size'],
        'clip_value': params['mapper_clip_value'],
        'temperature': params['mic_temperature'],
        'delta': params['mic_delta'],
        'hard_forward': params['mic_hard_forward'],
        'train_mode': params['mic_train_mode'],
        'power_constraint_mode': params['power_constraint_mode'],
    }


def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and 'model_state_dict' in checkpoint_obj:
        return checkpoint_obj['model_state_dict'], checkpoint_obj
    return checkpoint_obj, None


def strip_module_prefix(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}


def load_checkpoint_flexible(model, checkpoint_path, device='cpu', strict=False):
    checkpoint_obj = torch.load(checkpoint_path, map_location=device)
    state_dict, metadata = extract_state_dict(checkpoint_obj)
    load_result = unwrap_model(model).load_state_dict(strip_module_prefix(state_dict), strict=strict)
    return load_result, metadata


def apply_freeze_settings(model, params):
    model_ref = unwrap_model(model)

    for p in model_ref.encoder.parameters():
        p.requires_grad = not params['freeze_encoder']
    for p in model_ref.decoder.parameters():
        p.requires_grad = not params['freeze_decoder']
    if model_ref.mapper is not None:
        for p in model_ref.mapper.parameters():
            p.requires_grad = not params['freeze_mapper']


def build_optimizer(model, params):
    model_ref = unwrap_model(model)
    param_groups = []

    encoder_params = [p for p in model_ref.encoder.parameters() if p.requires_grad]
    decoder_params = [p for p in model_ref.decoder.parameters() if p.requires_grad]
    mapper_params = [p for p in model_ref.mapper.parameters() if p.requires_grad] if model_ref.mapper is not None else []

    if encoder_params:
        param_groups.append({'params': encoder_params, 'lr': params['init_lr'], 'group_name': 'encoder'})
    if decoder_params:
        param_groups.append({'params': decoder_params, 'lr': params['init_lr'], 'group_name': 'decoder'})
    if mapper_params:
        mapper_lr = params['mapper_lr'] if params['mapper_lr'] is not None else params['init_lr']
        param_groups.append({'params': mapper_params, 'lr': mapper_lr, 'group_name': 'mapper'})

    if not param_groups:
        raise RuntimeError('No trainable parameters found. Check freeze flags.')

    return optim.Adam(param_groups, weight_decay=params['weight_decay'])


def update_mapper_schedule(model, params, epoch):
    if not params.get('mic_anneal', False):
        return

    model_ref = unwrap_model(model)
    if model_ref.mapper is None or params.get('mapper_type', 'none') != 'mic':
        return

    steps = max(params['epochs'] - 1, 1)
    ratio = min(max(epoch / steps, 0.0), 1.0)

    if params.get('mic_delta') is None:
        temp_start = float(params['mic_temperature'])
        temp_end = float(params.get('mic_temperature_end', max(0.02, temp_start * 0.2)))
        temperature = temp_start + (temp_end - temp_start) * ratio
        model_ref.mapper.set_temperature(temperature)
    else:
        delta_start = float(params['mic_delta'])
        delta_end = float(params.get('mic_delta_end', max(delta_start, delta_start * 4.0)))
        delta = delta_start + (delta_end - delta_start) * ratio
        model_ref.mapper.set_delta(delta)


def aggregate_mapper_stats(aggregated, batch_stats):
    if not batch_stats:
        return

    aggregated['count'] += 1

    if 'usage_counts' in batch_stats:
        usage = batch_stats['usage_counts'].float()
        if aggregated['usage_counts'] is None:
            aggregated['usage_counts'] = usage.clone()
        else:
            aggregated['usage_counts'] += usage

    scalar_keys = [
        'usage_entropy',
        'active_fraction',
        'avg_nearest_distance',
        'mapper_output_power',
        'codebook_power',
        'min_interpoint_distance',
        'nearest_distance_mean',
        'nearest_distance_std',
        'nearest_distance_max',
    ]
    for key in scalar_keys:
        if key in batch_stats:
            aggregated[key] += float(batch_stats[key])


def finalize_mapper_stats(aggregated):
    count = max(aggregated['count'], 1)
    result = {}
    for key, val in aggregated.items():
        if key in {'count', 'usage_counts'}:
            continue
        result[key] = val / count

    usage_counts = aggregated['usage_counts']
    if usage_counts is not None:
        total = usage_counts.sum().item()
        probs = usage_counts / max(total, 1.0)
        mask = probs > 0
        entropy = -(probs[mask] * torch.log(probs[mask] + 1e-8)).sum().item() if mask.any() else 0.0
        result['usage_counts'] = usage_counts.cpu().tolist()
        result['usage_entropy_aggregated'] = entropy
        result['active_fraction_aggregated'] = float((usage_counts > 0).float().mean().item())
    return result


def maybe_export_constellation(model, params, export_base_path, epoch=None):
    model_ref = unwrap_model(model)
    if params.get('mapper_type', 'none') != 'mic' or model_ref.mapper is None:
        return None

    suffix = '' if epoch is None else '_epoch_{}'.format(epoch)
    export_path = export_base_path + suffix
    metadata = {
        'dataset': params['dataset'],
        'snr': params['snr'],
        'ratio': params['ratio'],
        'channel': params['channel'],
        'mapper_type': params['mapper_type'],
        'constellation_size': params['constellation_size'],
        'power_constraint_mode': params['power_constraint_mode'],
    }
    return model_ref.export_mapper_state(export_path, extra_metadata=metadata)


def train_epoch(model, optimizer, param, data_loader):
    model.train()
    epoch_loss = 0
    mapper_stats_agg = {
        'count': 0,
        'usage_counts': None,
        'usage_entropy': 0.0,
        'active_fraction': 0.0,
        'avg_nearest_distance': 0.0,
        'mapper_output_power': 0.0,
        'codebook_power': 0.0,
        'min_interpoint_distance': 0.0,
        'nearest_distance_mean': 0.0,
        'nearest_distance_std': 0.0,
        'nearest_distance_max': 0.0,
    }

    for iter, (images, _) in enumerate(data_loader):
        images = images.cuda() if param['parallel'] and torch.cuda.device_count(
        ) > 1 else images.to(param['device'])
        optimizer.zero_grad()
        outputs = model.forward(images)
        outputs = image_normalization('denormalization')(outputs)
        images = image_normalization('denormalization')(images)
        loss = model.loss(images, outputs) if not param['parallel'] else model.module.loss(
            images, outputs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()

        batch_mapper_stats = unwrap_model(model).get_mapper_stats()
        aggregate_mapper_stats(mapper_stats_agg, batch_mapper_stats)
    epoch_loss /= (iter + 1)

    return epoch_loss, optimizer, finalize_mapper_stats(mapper_stats_agg)


def evaluate_epoch(model, param, data_loader):
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for iter, (images, _) in enumerate(data_loader):
            images = images.cuda() if param['parallel'] and torch.cuda.device_count(
            ) > 1 else images.to(param['device'])
            outputs = model.forward(images)
            outputs = image_normalization('denormalization')(outputs)
            images = image_normalization('denormalization')(images)
            loss = model.loss(images, outputs) if not param['parallel'] else model.module.loss(
                images, outputs)
            epoch_loss += loss.detach().item()
        epoch_loss /= (iter + 1)

    return epoch_loss


def config_parser_pipeline():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'imagenet'], help='dataset')
    parser.add_argument('--out', default='./out', type=str, help='out_path')
    parser.add_argument('--disable_tqdm', default=False, type=str2bool, help='disable_tqdm')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--parallel', default=False, type=str2bool, help='parallel')
    parser.add_argument('--snr_list', default=['19', '13',
                        '7', '4', '1'], nargs='+', help='snr_list')
    parser.add_argument('--ratio_list', default=['1/6', '1/12'], nargs='+', help='ratio_list')
    parser.add_argument('--channel', default='AWGN', type=str,
                        choices=['AWGN', 'Rayleigh'], help='channel')

    parser.add_argument('--mapper_type', default='none', type=str,
                        choices=['none', 'mic'], help='constellation mapper type')
    parser.add_argument('--constellation_size', default=16, type=int,
                        help='number of constellation points for MIC')
    parser.add_argument('--mic_temperature', default=0.1, type=float,
                        help='MIC softmax temperature when delta is not set')
    parser.add_argument('--mic_delta', default=None, type=float,
                        help='MIC distance scaling factor for softmax(-delta * distance)')
    parser.add_argument('--mic_hard_forward', default=True, type=str2bool,
                        help='use hard-forward with soft surrogate gradients')
    parser.add_argument('--mic_train_mode', default='hard_forward_soft_backward', type=str,
                        choices=['soft', 'straight_through', 'hard_forward_soft_backward'],
                        help='MIC train mode')
    parser.add_argument('--mapper_clip_value', default=2.0, type=float,
                        help='symbol clipping range before mapper')
    parser.add_argument('--mapper_finetune_from', default='', type=str,
                        help='optional checkpoint to initialize from')
    parser.add_argument('--freeze_encoder', default=False, type=str2bool,
                        help='freeze encoder during fine-tuning')
    parser.add_argument('--freeze_decoder', default=False, type=str2bool,
                        help='freeze decoder during fine-tuning')
    parser.add_argument('--freeze_mapper', default=False, type=str2bool,
                        help='freeze mapper during fine-tuning')
    parser.add_argument('--mapper_lr', default=None, type=float,
                        help='optional mapper learning rate')
    parser.add_argument('--power_constraint_mode', default='codebook', type=str,
                        choices=['codebook', 'post_mapper', 'none'],
                        help='power normalization strategy for mapper')
    parser.add_argument('--export_constellation_path', default='', type=str,
                        help='optional base path for constellation export artifacts')

    parser.add_argument('--mic_anneal', default=False, type=str2bool,
                        help='linearly anneal MIC temperature or delta over training')
    parser.add_argument('--mic_temperature_end', default=0.02, type=float,
                        help='final temperature when annealing')
    parser.add_argument('--mic_delta_end', default=None, type=float,
                        help='final delta when annealing')

    parser.add_argument('--mrc_levels_per_axis', default=4, type=int,
                        help='reserved for future MRC support')
    parser.add_argument('--mrc_init_bounds', default='', type=str,
                        help='reserved for future MRC support')

    return parser.parse_args()


def main_pipeline():
    args = config_parser_pipeline()

    print("Training Start")
    dataset_name = args.dataset
    out_dir = args.out
    args.snr_list = list(map(float, args.snr_list))
    args.ratio_list = list(map(lambda x: float(Fraction(x)), args.ratio_list))
    params = {}
    params['disable_tqdm'] = args.disable_tqdm
    params['dataset'] = dataset_name
    params['out_dir'] = out_dir
    params['device'] = args.device
    params['snr_list'] = args.snr_list
    params['ratio_list'] = args.ratio_list
    params['channel'] = args.channel
    params['mapper_type'] = args.mapper_type
    params['constellation_size'] = args.constellation_size
    params['mic_temperature'] = args.mic_temperature
    params['mic_delta'] = args.mic_delta
    params['mic_hard_forward'] = args.mic_hard_forward
    params['mic_train_mode'] = args.mic_train_mode
    params['mapper_clip_value'] = args.mapper_clip_value
    params['mapper_finetune_from'] = args.mapper_finetune_from
    params['freeze_encoder'] = args.freeze_encoder
    params['freeze_decoder'] = args.freeze_decoder
    params['freeze_mapper'] = args.freeze_mapper
    params['mapper_lr'] = args.mapper_lr
    params['power_constraint_mode'] = args.power_constraint_mode
    params['export_constellation_path'] = args.export_constellation_path
    params['mic_anneal'] = args.mic_anneal
    params['mic_temperature_end'] = args.mic_temperature_end
    params['mic_delta_end'] = args.mic_delta_end
    params['mrc_levels_per_axis'] = args.mrc_levels_per_axis
    params['mrc_init_bounds'] = args.mrc_init_bounds
    if dataset_name == 'cifar10':
        params['batch_size'] = 64  # 1024
        params['num_workers'] = 4
        params['epochs'] = 1000
        params['init_lr'] = 1e-3  # 1e-2
        params['weight_decay'] = 5e-4
        params['parallel'] = False
        params['if_scheduler'] = True
        params['step_size'] = 640
        params['gamma'] = 0.1
        params['seed'] = 42
        params['ReduceLROnPlateau'] = False
        params['lr_reduce_factor'] = 0.5
        params['lr_schedule_patience'] = 15
        params['max_time'] = 12
        params['min_lr'] = 1e-5
    elif dataset_name == 'imagenet':
        params['batch_size'] = 32
        params['num_workers'] = 4
        params['epochs'] = 300
        params['init_lr'] = 1e-4
        params['weight_decay'] = 5e-4
        params['parallel'] = True
        params['if_scheduler'] = True
        params['gamma'] = 0.1
        params['seed'] = 42
        params['ReduceLROnPlateau'] = True
        params['lr_reduce_factor'] = 0.5
        params['lr_schedule_patience'] = 15
        params['max_time'] = 12
        params['min_lr'] = 1e-5
    else:
        raise Exception('Unknown dataset')

    set_seed(params['seed'])

    for ratio in params['ratio_list']:
        for snr in params['snr_list']:
            params['ratio'] = ratio
            params['snr'] = snr

            train_pipeline(params)


# add train_pipeline to with only dataset_name args
def train_pipeline(params):

    dataset_name = params['dataset']
    # load data
    if dataset_name == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), ])
        train_dataset = datasets.CIFAR10(root='../dataset/', train=True,
                                         download=True, transform=transform)

        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=params['batch_size'], num_workers=params['num_workers'])
        test_dataset = datasets.CIFAR10(root='../dataset/', train=False,
                                        download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=params['batch_size'], num_workers=params['num_workers'])

    elif dataset_name == 'imagenet':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((128, 128))])  # the size of paper is 128
        print("loading data of imagenet")
        train_dataset = datasets.ImageFolder(root='../dataset/ImageNet/train', transform=transform)

        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=params['batch_size'], num_workers=params['num_workers'])
        test_dataset = Vanilla(root='../dataset/ImageNet/val', transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=params['batch_size'], num_workers=params['num_workers'])
    else:
        raise Exception('Unknown dataset')

    # create model
    image_fisrt = train_dataset.__getitem__(0)[0]
    c = ratio2filtersize(image_fisrt, params['ratio'])
    print("The snr is {}, the inner channel is {}, the ratio is {:.2f}".format(
        params['snr'], c, params['ratio']))
    mapper_kwargs = get_mapper_kwargs_from_params(params)
    model = DeepJSCC(
        c=c,
        channel_type=params['channel'],
        snr=params['snr'],
        mapper_type=params['mapper_type'],
        mapper_kwargs=mapper_kwargs,
    )

    # init exp dir
    out_dir = params['out_dir']
    phaser = dataset_name.upper() + '_' + str(c) + '_' + str(params['snr']) + '_' + \
        "{:.2f}".format(params['ratio']) + '_' + str(params['channel']) + \
        '_' + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
    if params['mapper_type'] != 'none':
        phaser += '_{}_M{}'.format(params['mapper_type'].upper(), params['constellation_size'])
    params['phaser'] = phaser
    root_log_dir = out_dir + '/' + 'logs/' + phaser
    root_ckpt_dir = out_dir + '/' + 'checkpoints/' + phaser
    root_config_dir = out_dir + '/' + 'configs/' + phaser
    writer = SummaryWriter(log_dir=root_log_dir)

    # model init
    device = torch.device(params['device'] if torch.cuda.is_available() else 'cpu')
    if params['parallel'] and torch.cuda.device_count() > 1:
        model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model = model.cuda()
    else:
        model = model.to(device)

    if params['mapper_finetune_from']:
        load_result, _ = load_checkpoint_flexible(
            model,
            params['mapper_finetune_from'],
            device=device,
            strict=False,
        )
        print('Loaded finetune checkpoint: {}'.format(params['mapper_finetune_from']))
        print('Missing keys: {}'.format(load_result.missing_keys))
        print('Unexpected keys: {}'.format(load_result.unexpected_keys))

    model_ref = unwrap_model(model)
    if model_ref.mapper is not None and params['mapper_type'] == 'mic':
        model_ref.mapper.set_train_mode(params['mic_train_mode'])
        model_ref.mapper.hard_forward = params['mic_hard_forward']
        model_ref.mapper.set_temperature(params['mic_temperature'])
        model_ref.mapper.set_delta(params['mic_delta'])

    apply_freeze_settings(model, params)

    # opt
    optimizer = build_optimizer(model, params)
    if params['if_scheduler'] and not params['ReduceLROnPlateau']:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=params['step_size'], gamma=params['gamma'])
    elif params['ReduceLROnPlateau']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                         factor=params['lr_reduce_factor'],
                                                         patience=params['lr_schedule_patience'],
                                                         verbose=False)
    else:
        print("No scheduler")
        scheduler = None

    writer.add_text('config', str(params))
    t0 = time.time()
    epoch_train_losses, epoch_val_losses = [], []
    per_epoch_time = []

    # train
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        with tqdm(range(params['epochs']), disable=params['disable_tqdm']) as t:
            for epoch in t:

                t.set_description('Epoch %d' % epoch)

                start = time.time()
                update_mapper_schedule(model, params, epoch)

                epoch_train_loss, optimizer, epoch_mapper_stats = train_epoch(
                    model, optimizer, params, train_loader)

                epoch_val_loss = evaluate_epoch(model, params, test_loader)

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)

                writer.add_scalar('train/_loss', epoch_train_loss, epoch)
                writer.add_scalar('val/_loss', epoch_val_loss, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

                for key, val in epoch_mapper_stats.items():
                    if key == 'usage_counts':
                        continue
                    writer.add_scalar('mapper/{}'.format(key), val, epoch)

                if 'usage_counts' in epoch_mapper_stats:
                    writer.add_text('mapper/usage_counts', json.dumps(epoch_mapper_stats['usage_counts']), epoch)

                t.set_postfix(time=time.time() - start, lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss)

                per_epoch_time.append(time.time() - start)

                # Saving checkpoint

                if not os.path.exists(root_ckpt_dir):
                    os.makedirs(root_ckpt_dir)
                ckpt_path = '{}.pkl'.format(root_ckpt_dir + "/epoch_" + str(epoch))
                torch.save({
                    'model_state_dict': unwrap_model(model).state_dict(),
                    'epoch': epoch,
                    'inner_channel': c,
                    'params': params,
                    'mapper_config': unwrap_model(model).get_mapper_config(),
                }, ckpt_path)

                files = glob.glob(root_ckpt_dir + '/*.pkl')
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch - 1:
                        os.remove(file)

                if params['ReduceLROnPlateau'] and scheduler is not None:
                    scheduler.step(epoch_val_loss)
                elif params['if_scheduler'] and not params['ReduceLROnPlateau']:
                    scheduler.step()  # use only information from the validation loss

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

                # Stop training after params['max_time'] hours
                if time.time() - t0 > params['max_time'] * 3600:
                    print('-' * 89)
                    print("Max_time for training elapsed {:.2f} hours, so stopping".format(
                        params['max_time']))
                    break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early because of KeyboardInterrupt')

    test_loss = evaluate_epoch(model, params, test_loader)
    train_loss = evaluate_epoch(model, params, train_loader)
    print("Test Accuracy: {:.4f}".format(test_loss))
    print("Train Accuracy: {:.4f}".format(train_loss))
    print("Convergence Time (Epochs): {:.4f}".format(epoch))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time() - t0))
    print("AVG TIME PER EPOCH: {:.4f}s".format(np.mean(per_epoch_time)))

    if params.get('mapper_type', 'none') == 'mic':
        if params.get('export_constellation_path'):
            export_base = os.path.join(params['export_constellation_path'], params['phaser'])
        else:
            export_base = os.path.join(root_ckpt_dir, 'constellation')
        export_info = maybe_export_constellation(model, params, export_base_path=export_base, epoch=None)
        if export_info is not None:
            print('Exported constellation artifacts: {}'.format(export_info))

    """
        Write the results in out_dir/results folder
    """

    writer.add_text(tag='result', text_string="""Dataset: {}\nparams={}\n\nTotal Parameters: {}\n\n
    FINAL RESULTS\nTEST Loss: {:.4f}\nTRAIN Loss: {:.4f}\n\n
    Convergence Time (Epochs): {:.4f}\nTotal Time Taken: {:.4f} hrs\nAverage Time Per Epoch: {:.4f} s\n\n\n"""
                    .format(dataset_name, params, view_model_param(model), np.mean(np.array(train_loss)),
                            np.mean(np.array(test_loss)), epoch, (time.time() - t0) / 3600, np.mean(per_epoch_time)))
    writer.close()
    if not os.path.exists(os.path.dirname(root_config_dir)):
        os.makedirs(os.path.dirname(root_config_dir))
    with open(root_config_dir + '.yaml', 'w') as f:
        dict_yaml = {'dataset_name': dataset_name, 'params': params,
                     'inner_channel': c, 'total_parameters': view_model_param(model)}
        import yaml
        yaml.dump(dict_yaml, f)

    del model, optimizer, scheduler, train_loader, test_loader
    del writer


def train(args, ratio: float, snr: float):  # deprecated

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    # load data
    if args.dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), ])
        train_dataset = datasets.CIFAR10(root='../dataset/', train=True,
                                         download=True, transform=transform)

        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=args.batch_size, num_workers=args.num_workers)
        test_dataset = datasets.CIFAR10(root='../dataset/', train=False,
                                        download=True, transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=args.batch_size, num_workers=args.num_workers)
    elif args.dataset == 'imagenet':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((128, 128))])  # the size of paper is 128
        print("loading data of imagenet")
        train_dataset = datasets.ImageFolder(root='./dataset/ImageNet/train', transform=transform)

        train_loader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=args.batch_size, num_workers=args.num_workers)
        test_dataset = Vanilla(root='./dataset/ImageNet/val', transform=transform)
        test_loader = DataLoader(test_dataset, shuffle=True,
                                 batch_size=args.batch_size, num_workers=args.num_workers)
    else:
        raise Exception('Unknown dataset')

    print(args)
    image_fisrt = train_dataset.__getitem__(0)[0]
    c = ratio2filtersize(image_fisrt, ratio)
    print("the inner channel is {}".format(c))
    model = DeepJSCC(c=c, channel_type=args.channel, snr=snr)

    if args.parallel and torch.cuda.device_count() > 1:
        model = DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model = model.cuda()
        criterion = nn.MSELoss(reduction='mean').cuda()
    else:
        model = model.to(device)
        criterion = nn.MSELoss(reduction='mean').to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.if_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    epoch_loop = tqdm(range(args.epochs), total=args.epochs, leave=True, disable=args.disable_tqdm)
    for epoch in epoch_loop:
        run_loss = 0.0
        for images, _ in tqdm((train_loader), leave=False, disable=args.disable_tqdm):
            optimizer.zero_grad()
            images = images.cuda() if args.parallel and torch.cuda.device_count() > 1 else images.to(device)
            outputs = model(images)
            outputs = image_normalization('denormalization')(outputs)
            images = image_normalization('denormalization')(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
        if args.if_scheduler:  # the scheduler is wrong before
            scheduler.step()
        with torch.no_grad():
            model.eval()
            test_mse = 0.0
            for images, _ in tqdm((test_loader), leave=False, disable=args.disable_tqdm):
                images = images.cuda() if args.parallel and torch.cuda.device_count() > 1 else images.to(device)
                outputs = model(images)
                images = image_normalization('denormalization')(images)
                outputs = image_normalization('denormalization')(outputs)
                loss = criterion(outputs, images)
                test_mse += loss.item()
            model.train()
        # epoch_loop.set_postfix(loss=run_loss/len(train_loader), test_mse=test_mse/len(test_loader))
        print("epoch: {}, loss: {:.4f}, test_mse: {:.4f}, lr:{}".format(
            epoch, run_loss / len(train_loader), test_mse / len(test_loader), optimizer.param_groups[0]['lr']))
    save_model(model, args.saved, args.saved + '/{}_{}_{:.2f}_{:.2f}_{}_{}.pth'
               .format(args.dataset, args.epochs, ratio, snr, args.batch_size, c))


def config_parser():  # deprecated
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=2048, type=int, help='Random seed')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--epochs', default=256, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--channel', default='AWGN', type=str,
                        choices=['AWGN', 'Rayleigh'], help='channel type')
    parser.add_argument('--saved', default='./saved', type=str, help='saved_path')
    parser.add_argument('--snr_list', default=['19', '13',
                        '7', '4', '1'], nargs='+', help='snr_list')
    parser.add_argument('--ratio_list', default=['1/3',
                        '1/6', '1/12'], nargs='+', help='ratio_list')
    parser.add_argument('--num_workers', default=0, type=int, help='num_workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'imagenet'], help='dataset')
    parser.add_argument('--parallel', default=False, type=bool, help='parallel')
    parser.add_argument('--if_scheduler', default=False, type=bool, help='if_scheduler')
    parser.add_argument('--step_size', default=640, type=int, help='scheduler')
    parser.add_argument('--device', default='cuda:0', type=str, help='device')
    parser.add_argument('--gamma', default=0.5, type=float, help='gamma')
    parser.add_argument('--disable_tqdm', default=True, type=bool, help='disable_tqdm')
    return parser.parse_args()


def main():  # deprecated
    args = config_parser()
    args.snr_list = list(map(float, args.snr_list))
    args.ratio_list = list(map(lambda x: float(Fraction(x)), args.ratio_list))
    set_seed(args.seed)
    print("Training Start")
    for ratio in args.ratio_list:
        for snr in args.snr_list:
            train(args, ratio, snr)


if __name__ == '__main__':
    main_pipeline()
    # main()
    # main()
