import os
from tqdm import tqdm
import torch
import numpy as np
import dataset
from generator import Generator
from encoder_net import EncoderNet
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from utils.visualizer import save_image
from models.perceptual_model import PerceptualModel


def main():
  generator = Generator()
  encoder = EncoderNet().to('cuda')
  feat_model = PerceptualModel(min_val=generator.G.min_val, 
                             max_val=generator.G.max_val)
  imgTrainDataset = dataset.trainImgDataset()
  dataloader = DataLoader(imgTrainDataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=True)
  optimizer = optim.SGD(encoder.parameters(),
                        lr=0.001,
                        momentum=0.9)
  scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                        step_size=10, 
                                        gamma=0.1)

  num_epochs = 30
  feat_lambda = 1.0

  def compute_loss(imgs):
      imgs = imgs.to('cuda')
      wps = encoder(imgs)
      reimgs = generator.generate_timgs_from_wps(wps)
      img_feats = feat_model.net(imgs)
      reimgs_feats = feat_model.net(reimgs)
      imgloss = F.mse_loss(imgs, reimgs, 
                           reduction='mean')
      featloss = F.mse_loss(img_feats, reimgs_feats,
                            reduction='mean')
      loss = imgloss + feat_lambda * featloss
      return loss

  def train_loop():
    for imgs in tqdm(dataloader):
      optimizer.zero_grad()
      loss = compute_loss(imgs)
      loss.backward()
      optimizer.step()
    return loss.to('cpu').detach().numpy()

  for epoch in range(num_epochs):
    loss = train_loop()
    print(f'{epoch}: loss={loss}')
    scheduler.step()

  torch.save(encoder.state_dict(), 'encoder_weights2.pth')


def next_img_name(path='.'):
  imgfiles = [imgfile for imgfile in os.listdir(path) 
              if imgfile.split('.')[0].isdecimal()]
  maxnum = int(max(imgfiles).split('.')[0]) if len(imgfiles) else 0
  return str(maxnum + 1) + '.png'


if __name__ == "__main__":
  main()