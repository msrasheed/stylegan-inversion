import torch
from encoder_net import EncoderNet
from models.stylegan_encoder import StyleGANEncoder

class Encoder:
  def __init__(self, use_fallback=False,
               weights_path=None,
               train=False):
    self.use_fallback = use_fallback
    self.weights_path = weights_path
    self.train = train
    self.encoder = self.getImgEncoder()

  def getImgEncoder(self):
    if self.use_fallback:
      encoder = StyleGANEncoder('styleganinv_ffhq256', None)
      return encoder
    else:
      encoder = EncoderNet()
      if self.weights_path is not None:
        encoder.load_state_dict(torch.load(self.weights_path))
      encoder.to('cuda')
      if not self.train:
        encoder.eval()
      return encoder

  def getEncoderWp(self, imgs):
    if self.use_fallback:
      return self.encoder.net(imgs).view(-1, 14, 512)
    else:
      return self.encoder(imgs)

  def parameters(self):
    if self.use_fallback:
      return self.encoder.net.parameters()
    return self.encoder.parameters()

  def state_dict(self):
    if self.use_fallback:
      return self.encoder.net.state_dict()
    return self.encoder.state_dict()
  
  def __call__(self, imgs):
    return self.getEncoderWp(imgs)