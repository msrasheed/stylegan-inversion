import os
from tqdm import tqdm
import torch
import numpy as np
from encoder_net import EncoderNet
import dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from utils.visualizer import save_image
from models.perceptual_model import PerceptualModel
from models.stylegan_encoder import StyleGANEncoder
import text_encoder

USE_MY_ENCODER = True

class FallbackEncoder:
  def __init__(self, myImgEncoder=True):
    self.myImgEncoder = myImgEncoder
    self.encoder = self.getImgEncoder()

  def getImgEncoder(self):
    if self.myImgEncoder:
      encoder = EncoderNet()
      encoder.load_state_dict(torch.load('encoder_weights.pth'))
      encoder.to('cuda')
      encoder.eval()
      return encoder
    else:
      encoder = StyleGANEncoder('styleganinv_ffhq256', None)
      return encoder

  def getEncoderWp(self, imgs):
    if self.myImgEncoder:
      return self.encoder(imgs)
    else:
      return self.encoder.net(imgs).view(1, 14, 512)
  
  def __call__(self, imgs):
    return self.getEncoderWp(imgs)

def main():
  textenc = text_encoder.TextEncoder()
  encoder = FallbackEncoder(USE_MY_ENCODER)
  encoder.load_state_dict(torch.load('encoder_weights.pth'))
  encoder.to('cuda')
  encoder.eval()
  textTrainDataset = dataset.CelebAHQTextDataset()
  dataloader = DataLoader(textTrainDataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=True)
  optimizer = optim.SGD(textenc.parameters(),
                        lr=0.01,
                        momentum=0.9)
  scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                        step_size=30, 
                                        gamma=0.1)

  num_epochs = 30

  def compute_loss(descs, imgs):
      descs = descs.to('cuda')
      imgs = imgs.to('cuda')
      imgs_wps = encoder(imgs)
      descs_wps = textenc(descs)
      loss = F.mse_loss(imgs_wps, descs_wps, 
                           reduction='mean')
      return loss

  def train_loop():
    for descs, imgs in tqdm(dataloader):
      optimizer.zero_grad()
      loss = compute_loss(descs, imgs)
      loss.backward()
      optimizer.step()
    return loss.to('cpu').detach().numpy()

  for epoch in range(num_epochs):
    loss = train_loop()
    print(f'{epoch}: loss={loss}')
    scheduler.step()

  torch.save(textenc.state_dict(), 'text_encoder_weights.pth')


def next_img_name(path='.'):
  imgfiles = [imgfile for imgfile in os.listdir(path) 
              if imgfile.split('.')[0].isdecimal()]
  maxnum = int(max(imgfiles).split('.')[0]) if len(imgfiles) else 0
  return str(maxnum + 1) + '.png'


if __name__ == "__main__":
  main()