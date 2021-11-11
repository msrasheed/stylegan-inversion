import os
import sys
import torch
import dataset
import numpy as np
from tqdm import tqdm
from models.stylegan_generator_network import StyleGANGeneratorNet
from encoder import Encoder
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from utils.visualizer import save_image
from models.perceptual_model import VGG16
from signal import signal, SIGINT

def loadStyleGan():
  return StyleGAN()

class StyleGAN:
  def __init__(self):
    generator = StyleGANGeneratorNet(resolution=256,
                                     repeat_w=False,
                                     final_tanh=True)
    state_dict = torch.load('models/pretrain/styleganinv_ffhq256_generator.pth')
    state_dict['truncation.truncation'] = generator.state_dict()['truncation.truncation']
    generator.load_state_dict(state_dict)
    generator.to('cuda')
    generator.eval()
    for param in generator.parameters():
      param.requires_grad = False
    self.generator = generator
  def __call__(self, x):
   return self.generator.synthesis(x)

def loadFeatureNet():
  model = VGG16()
  model.load_state_dict(torch.load('models/pretrain/vgg16.pth'))
  model.to('cuda')
  model.eval()
  for param in model.parameters():
    param.requires_grad = False
  return model


def main():
  generator = loadStyleGan()
  encoder = Encoder(use_fallback=False,
                    weights_path='encoder_weights_pretrain2.pth',
                    train=True)
  feat_model = loadFeatureNet()
  imgTrainDataset = dataset.trainImgDataset()
  dataloader = DataLoader(imgTrainDataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=True)

  num_epochs = 30
  feat_lambda = 1.0

  def intHandler(sigNum, frame):
    weight_name = 'encoder_weights_test.pth'
    save_weights = input("\nsave weights? y/[n]:")
    if save_weights == 'y':
      torch.save(encoder.state_dict(), weight_name)
      print(f"saved weights to {weight_name}")
    exit(0)
  signal(SIGINT, intHandler)

  def compute_loss(imgs):
      imgs = imgs.to('cuda')
      wps = encoder(imgs).view(-1, 14, 512)
      reimgs = generator(wps)
      img_feats = feat_model(imgs)
      reimgs_feats = feat_model(reimgs)
      imgloss = F.mse_loss(imgs, reimgs, 
                           reduction='mean')
      featloss = F.mse_loss(img_feats, reimgs_feats,
                            reduction='mean')
      loss = imgloss + feat_lambda * featloss
      return loss

  def train_loop():
    losshist = np.zeros(len(dataloader))
    for i, imgs in enumerate(tqdm(dataloader)):
      optimizer.zero_grad()
      loss = compute_loss(imgs)
      loss.backward()
      optimizer.step()
      losshist[i] = loss.cpu().detach().numpy()
    return np.mean(losshist)


  optimizer = optim.SGD(encoder.parameters(),
                        lr=0.00001, momentum=0.9)
  scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                        step_size=10, 
                                        gamma=0.1)
  for epoch in range(num_epochs):
    loss = train_loop()
    print(f'{epoch+15}: loss={loss}')
    scheduler.step()

  torch.save(encoder.state_dict(), 'encoder_weights_2.pth')


def next_img_name(path='.'):
  imgfiles = [imgfile for imgfile in os.listdir(path) 
              if imgfile.split('.')[0].isdecimal()]
  maxnum = int(max(imgfiles).split('.')[0]) if len(imgfiles) else 0
  return str(maxnum + 1) + '.png'


if __name__ == "__main__":
  main()
