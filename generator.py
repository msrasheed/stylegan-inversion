import cv2
import torch
import numpy as np
from models.stylegan_generator import StyleGANGenerator
from models.perceptual_model import PerceptualModel
from utils.visualizer import save_image, load_image, resize_image

def _get_tensor_value(tensor):
  """Gets the value of a torch Tensor."""
  return tensor.cpu().detach().numpy()
  
class Generator:
  def __init__(self):
    self.model_name = 'styleganinv_ffhq256'
    self.logger = None
    self.G = StyleGANGenerator(self.model_name, self.logger)
    self.F = PerceptualModel(min_val=self.G.min_val, max_val=self.G.max_val)
    self.run_device = self.G.run_device

  def generate_img(self):
    init_z = self.G.sample(1, latent_space_type='wp',
                           z_space_dim=512, num_layers=14)
    init_z = self.G.preprocess(init_z, latent_space_type='wp')
    x = self.G._synthesize(init_z, latent_space_type='wp')['image']

    viz_results = []
    viz_results.append(self.G.postprocess(x)[0])

    return viz_results

  def generate_timg_from_wp(self, wp):
    wp = wp.view(1, 14, 512).numpy()
    x = self.G._synthesize(wp, latent_space_type='wp')['image']
    x = torch.Tensor(x).to(self.run_device)
    return x

  def generate_timgs_from_wp(self, wp):
    wp = wp.view(1, 14, 512).numpy()
    x = self.G._synthesize(wp, latent_space_type='wp')['image']
    x = torch.Tensor(x).to(self.run_device)
    return x

  def generate_timgs_from_wps(self, wps):
    wps = wps.view(-1, 14, 512).numpy()
    xs = self.G.synthesize(wps, latent_space_type='wp')['image']
    return torch.Tensor(xs).to(self.run_device)

  def preprocess(self, image):
    """Preprocesses a single image.

    This function assumes the input numpy array is with shape [height, width,
    channel], channel order `RGB`, and pixel range [0, 255].

    The returned image is with shape [channel, new_height, new_width], where
    `new_height` and `new_width` are specified by the given generative model.
    The channel order of returned image is also specified by the generative
    model. The pixel range is shifted to [min_val, max_val], where `min_val` and
    `max_val` are also specified by the generative model.
    """
    if not isinstance(image, np.ndarray):
      raise ValueError(f'Input image should be with type `numpy.ndarray`!')
    if image.dtype != np.uint8:
      raise ValueError(f'Input image should be with dtype `numpy.uint8`!')

    if image.ndim != 3 or image.shape[2] not in [1, 3]:
      raise ValueError(f'Input should be with shape [height, width, channel], '
                       f'where channel equals to 1 or 3!\n'
                       f'But {image.shape} is received!')
    if image.shape[2] == 1 and self.G.image_channels == 3:
      image = np.tile(image, (1, 1, 3))
    if image.shape[2] != self.G.image_channels:
      raise ValueError(f'Number of channels of input image, which is '
                       f'{image.shape[2]}, is not supported by the current '
                       f'inverter, which requires {self.G.image_channels} '
                       f'channels!')

    if self.G.image_channels == 3 and self.G.channel_order == 'BGR':
      image = image[:, :, ::-1]
    if image.shape[1:3] != [self.G.resolution, self.G.resolution]:
      image = cv2.resize(image, (self.G.resolution, self.G.resolution))
    image = image.astype(np.float32)
    image = image / 255.0 * (self.G.max_val - self.G.min_val) + self.G.min_val
    image = image.astype(np.float32).transpose(2, 0, 1)

    return image