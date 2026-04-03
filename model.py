# -*- coding: utf-8 -*-
"""
Created on Tue Dec  11:00:00 2023

@author: chun
"""

import torch
import torch.nn as nn

from channel import Channel
from constellation import MICLayer

""" def _image_normalization(norm_type):
    def _inner(tensor: torch.Tensor):
        if norm_type == 'nomalization':
            return tensor / 255.0
        elif norm_type == 'denormalization':
            return (tensor * 255.0).type(torch.FloatTensor)
        else:
            raise Exception('Unknown type of normalization')
    return _inner """


def ratio2filtersize(x: torch.Tensor, ratio):
    if x.dim() == 4:
        # before_size = np.prod(x.size()[1:])
        before_size = torch.prod(torch.tensor(x.size()[1:]))
    elif x.dim() == 3:
        # before_size = np.prod(x.size())
        before_size = torch.prod(torch.tensor(x.size()))
    else:
        raise Exception('Unknown size of input')
    encoder_temp = _Encoder(is_temp=True)
    z_temp = encoder_temp(x)
    # c = before_size * ratio / np.prod(z_temp.size()[-2:])
    c = before_size * ratio / torch.prod(torch.tensor(z_temp.size()[-2:]))
    return int(c)


class _ConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(_ConvWithPReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.prelu = nn.PReLU()

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.prelu(x)
        return x


class _TransConvWithPReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activate=nn.PReLU(), padding=0, output_padding=0):
        super(_TransConvWithPReLU, self).__init__()
        self.transconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.activate = activate
        if activate == nn.PReLU():
            nn.init.kaiming_normal_(self.transconv.weight, mode='fan_out',
                                    nonlinearity='leaky_relu')
        else:
            nn.init.xavier_normal_(self.transconv.weight)

    def forward(self, x):
        x = self.transconv(x)
        x = self.activate(x)
        return x


class _Encoder(nn.Module):
    def __init__(self, c=1, is_temp=False, P=1):
        super(_Encoder, self).__init__()
        self.is_temp = is_temp
        # self.imgae_normalization = _image_normalization(norm_type='nomalization')
        self.conv1 = _ConvWithPReLU(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv2 = _ConvWithPReLU(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv3 = _ConvWithPReLU(in_channels=32, out_channels=32,
                                    kernel_size=5, padding=2)  # padding size could be changed here
        self.conv4 = _ConvWithPReLU(in_channels=32, out_channels=32, kernel_size=5, padding=2)
        self.conv5 = _ConvWithPReLU(in_channels=32, out_channels=2*c, kernel_size=5, padding=2)
        self.norm = self._normlizationLayer(P=P)

    @staticmethod
    def _normlizationLayer(P=1):
        def _inner(z_hat: torch.Tensor):
            if z_hat.dim() == 4:
                batch_size = z_hat.size()[0]
                # k = np.prod(z_hat.size()[1:])
                k = torch.prod(torch.tensor(z_hat.size()[1:]))
            elif z_hat.dim() == 3:
                batch_size = 1
                # k = np.prod(z_hat.size())
                k = torch.prod(torch.tensor(z_hat.size()))
            else:
                raise Exception('Unknown size of input')
            # k = torch.tensor(k)
            z_temp = z_hat.reshape(batch_size, 1, 1, -1)
            z_trans = z_hat.reshape(batch_size, 1, -1, 1)
            tensor = torch.sqrt(P * k) * z_hat / torch.sqrt((z_temp @ z_trans))
            if batch_size == 1:
                return tensor.squeeze(0)
            return tensor
        return _inner

    def forward(self, x):
        # x = self.imgae_normalization(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        if not self.is_temp:
            x = self.conv5(x)
            x = self.norm(x)
        return x


class _Decoder(nn.Module):
    def __init__(self, c=1):
        super(_Decoder, self).__init__()
        # self.imgae_normalization = _image_normalization(norm_type='denormalization')
        self.tconv1 = _TransConvWithPReLU(
            in_channels=2*c, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv2 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv3 = _TransConvWithPReLU(
            in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.tconv4 = _TransConvWithPReLU(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.tconv5 = _TransConvWithPReLU(
            in_channels=16, out_channels=3, kernel_size=5, stride=2, padding=2, output_padding=1,activate=nn.Sigmoid())
        # may be some problems in tconv4 and tconv5, the kernal_size is not the same as the paper which is 5

    def forward(self, x):
        x = self.tconv1(x)
        x = self.tconv2(x)
        x = self.tconv3(x)
        x = self.tconv4(x)
        x = self.tconv5(x)
        # x = self.imgae_normalization(x)
        return x


class DeepJSCC(nn.Module):
    def __init__(self, c, channel_type='AWGN', snr=None, mapper_type='none', mapper_kwargs=None):
        super(DeepJSCC, self).__init__()
        self.encoder = _Encoder(c=c)
        self.mapper = None
        self.mapper_type = 'none'
        self.set_mapper(mapper_type=mapper_type, mapper_kwargs=mapper_kwargs)
        if snr is not None:
            self.channel = Channel(channel_type, snr)
        else:
            self.channel = None
        self.decoder = _Decoder(c=c)

    def forward(self, x):
        z = self.encoder(x)
        if self.mapper is not None:
            z = self.mapper(z)
        if hasattr(self, 'channel') and self.channel is not None:
            z = self.channel(z)
        x_hat = self.decoder(z)
        return x_hat

    def forward_debug(self, x, return_mapper_indices=False):
        """Debug forward pass exposing pre/post mapper tensors and reconstruction."""
        z_pre_mapper = self.encoder(x)

        mapper_indices = None
        if self.mapper is not None:
            if return_mapper_indices:
                z_post_mapper, mapper_indices = self.mapper(z_pre_mapper, return_indices=True)
            else:
                z_post_mapper = self.mapper(z_pre_mapper)
        else:
            z_post_mapper = z_pre_mapper

        if hasattr(self, 'channel') and self.channel is not None:
            z_after_channel = self.channel(z_post_mapper)
        else:
            z_after_channel = z_post_mapper

        x_hat = self.decoder(z_after_channel)

        out = {
            'z_pre_mapper': z_pre_mapper,
            'z_post_mapper': z_post_mapper,
            'z_after_channel': z_after_channel,
            'x_hat': x_hat,
        }
        if mapper_indices is not None:
            out['mapper_indices'] = mapper_indices
        return out

    def set_mapper(self, mapper_type='none', mapper_kwargs=None):
        mapper_kwargs = {} if mapper_kwargs is None else dict(mapper_kwargs)

        if mapper_type is None or str(mapper_type).lower() == 'none':
            self.disable_mapper()
            return

        mapper_type = str(mapper_type).lower()
        if mapper_type == 'mic':
            self.mapper = MICLayer(**mapper_kwargs)
            self.mapper_type = 'mic'
            return

        raise ValueError('Unknown mapper type: {}'.format(mapper_type))

    def disable_mapper(self):
        self.mapper = None
        self.mapper_type = 'none'

    def set_mapper_deploy_mode(self, enabled=True):
        if self.mapper is not None and hasattr(self.mapper, 'set_deploy_mode'):
            self.mapper.set_deploy_mode(enabled)

    def export_mapper_state(self, path, extra_metadata=None):
        if self.mapper is None:
            return None
        if hasattr(self.mapper, 'export_constellation'):
            return self.mapper.export_constellation(path, extra_metadata=extra_metadata)
        return None

    def get_mapper_stats(self):
        if self.mapper is not None and hasattr(self.mapper, 'get_stats'):
            return self.mapper.get_stats()
        return {}

    def get_mapper_config(self):
        if self.mapper_type == 'none' or self.mapper is None:
            return {'mapper_type': 'none'}

        config = {
            'mapper_type': self.mapper_type,
            'constellation_size': getattr(self.mapper, 'constellation_size', None),
            'clip_value': getattr(self.mapper, 'clip_value', None),
            'temperature': getattr(self.mapper, 'temperature', None),
            'delta': getattr(self.mapper, 'delta', None),
            'hard_forward': getattr(self.mapper, 'hard_forward', None),
            'train_mode': getattr(self.mapper, 'train_mode', None),
            'power_constraint_mode': getattr(self.mapper, 'power_constraint_mode', None),
        }
        return config

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None

    def loss(self, prd, gt):
        criterion = nn.MSELoss(reduction='mean')
        loss = criterion(prd, gt)
        return loss


if __name__ == '__main__':
    model = DeepJSCC(c=20)
    print(model)
    x = torch.rand(1, 3, 128, 128)
    y = model(x)
    print(y.size())
    print(y)
    print(model.encoder.norm)
    print(model.encoder.norm(y))
    print(model.encoder.norm(y).size())
    print(model.encoder.norm(y).size()[1:])
