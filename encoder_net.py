import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderNet(nn.Module):
  def __init__(self):
    super().__init__()

    self.num_blocks = 8 #log2(256) = 8
    self.encoder_channels_max = 1024 
    self.init_res = 4
    self.image_channels = 3
    self.resolution = 256
    self.w_space_dim = 512

    in_channels = 3
    out_channels = 64
    for block_idx in range(self.num_blocks):
      if block_idx == 0:
        self.add_module(
          f'block{block_idx}',
          FirstBlock(in_channels=in_channels,
                     out_channels=out_channels))

      elif block_idx == self.num_blocks - 1:
        in_channels = in_channels * self.init_res * self.init_res
        out_channels = self.w_space_dim * 2 * block_idx
        self.add_module(
          f'block{block_idx}',
          LastBlock(in_channels=in_channels,
                    out_channels=out_channels))

      else:
        self.add_module(
          f'block{block_idx}',
          ResBlock(in_channels=in_channels,
                   out_channels=out_channels))
      in_channels = out_channels
      out_channels = min(out_channels * 2, self.encoder_channels_max)

    self.downsample = AveragePoolingLayer()

  def forward(self, x):
    if x.ndim != 4 or x.shape[1:] != (
        self.image_channels, self.resolution, self.resolution):
      raise ValueError(f'The input image should be with shape [batch_size, '
                       f'channel, height, width], where '
                       f'`channel` equals to {self.image_channels}, '
                       f'`height` and `width` equal to {self.resolution}!\n'
                       f'But {x.shape} is received!')

    for block_idx in range(self.num_blocks):
      if 0 < block_idx < self.num_blocks - 1:
        x = self.downsample(x)
      x = self.__getattr__(f'block{block_idx}')(x)
    return x

class AveragePoolingLayer(nn.Module):
  def __init__(self, scale_factor=2):
    super().__init__()
    self.scale_factor = scale_factor

  def forward(self, x):
    ksize = [self.scale_factor, self.scale_factor]
    strides = [self.scale_factor, self.scale_factor]
    return F.avg_pool2d(x, kernel_size=ksize, stride=strides, padding=0)

class FirstBlock(nn.Module):
  def __init__(self,
               in_channels=3,
               out_channels=64):
    super().__init__()
    self.conv = nn.Conv2d(in_channels=3,
                          out_channels=64,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False)
    self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

  def forward(self, x):
    return self.activate(self.conv(x))
    
class WScaleLayer(nn.Module):
  def __init__(self,
               in_channels,
               out_channels,
               kernel_size,
               gain=np.sqrt(2.0)):
    super().__init__()
    # fan_in = in_channels * kernel_size * kernel_size
    # self.scale = gain / np.sqrt(fan_in)
    self.scale = 1.0
    self.bias = nn.Parameter(torch.zeros(out_channels))

  def forward(self, x):
    if x.ndim == 4:
      return x * self.scale + self.bias.view(1, -1, 1, 1)
    if x.ndim == 2:
      return x * self.scale + self.bias.view(1, -1)
    raise ValueError(f'The input tensor should be with shape [batch_size, '
                     f'channel, height, width], or [batch_size, channel]!\n'
                     f'But {x.shape} is received!')

class ResBlock(nn.Module):
  def __init__(self,
               in_channels,
               out_channels):
    super().__init__()
    # wscale_gain =  np.sqrt(2.0)
    # scale nor batch normalization used
    # WScaleLayers just to add/learn bias?

    if in_channels != out_channels:
      self.add_shortcut = True
      self.conv = nn.Conv2d(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False) 
    else:
      self.add_shortcut = False
      self.identity = nn.Identity()

    hidden_channels = min(in_channels, out_channels)

    self.conv1 = nn.Conv2d(in_channels=in_channels,
                          out_channels=hidden_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False)
    # self.scale1 = wscale_gain / np.sqrt(in_channels * 3 * 3)
    self.wscale1 = WScaleLayer(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=3,
                               gain=1.0)
    # self.bn1 = nn.Identity() 

    self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                           out_channels=out_channels,
                           kernel_size=3,
                           stride=1,
                           padding=1,
                           bias=False)
    # self.scale2 = wscale_gain / np.sqrt(hidden_channels * 3 * 3)
    self.wscale2 = WScaleLayer(in_channels=hidden_channels,
                               out_channels=out_channels,
                               kernel_size=3,
                               gain=1.0)
    # self.bn2 = nn.Identity()

    self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

  def forward(self, x):
    if self.add_shortcut:
      y = self.activate(self.conv(x))
    else:
      y= self.identity(x)
    x = self.activate(self.wscale1(self.conv1(x)))
    x = self.activate(self.wscale2(self.conv2(x)))
    return x + y

class LastBlock(nn.Module):
  def __init__(self,
               in_channels,
               out_channels):
    super().__init__()

    self.fc = nn.Linear(in_features=in_channels,
                        out_features=out_channels,
                        bias=False)
    self.scale = 1.0 / np.sqrt(in_channels)

  def forward(self, x):
    x = x.view(x.shape[0], -1)
    x = self.fc(x) * self.scale
    x = x.view(x.shape[0], x.shape[1], 1, 1)
    return x.view(x.shape[0], x.shape[1])