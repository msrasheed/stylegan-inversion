import os
import numpy as np
from generator import Generator
from encoder_net import EncoderNet
from dataset import CelebAHQImgDataset
from torch.utils.data import DataLoader
from torch.optim import optim
import torch.nn.functional as F
from utils.visualizer import save_image
from models.perceptual_model import PerceptualModel


def main():
  generator = Generator()
  encoder = EncoderNet()
  feat_model = PerceptualModel(min_val=generator.G.min_val, 
                             max_val=generator.G.max_val)
  imgTrainDataset = CelebAHQImgDataset()
  dataloader = DataLoader(imgTrainDataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=True)
  optimizer = optim.SGD(encoder.parameters(),
                        lr=0.01,
                        momentum=0.9)
  scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                        step_size=30, 
                                        gamma=0.1)

  num_epochs = 30
  loss = np.zeros(num_epochs)

  def train_loop():
    for imgs in dataloader:
      wps = encoder(imgs)
      reimgs = generator.generate_timgs_from_wps(wps)
      img_feats = feat_model.net(imgs)
      reimgs_feats = feat_model.net(reimgs)
      imgloss = F.mse_loss(imgs, reimgs, 
                           reduction='mean')
      featloss = F.mse_loss(img_feats, reimgs_feats,
                            reduction='mean')
      loss = imgloss + featloss


  for epoch in range(num_epochs):
    loss[epoch] = train_loop()


def next_img_name(path='.'):
  imgfiles = [imgfile for imgfile in os.listdir(path) 
              if imgfile.split('.')[0].isdecimal()]
  maxnum = int(max(imgfiles).split('.')[0]) if len(imgfiles) else 0
  return str(maxnum + 1) + '.png'


if __name__ == "__main__":
  main()