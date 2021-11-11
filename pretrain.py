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
from models.perceptual_model import VGG16
from signal import signal, SIGINT
import sys


def main():
  encoder = EncoderNet().to('cuda')
  imgTrainDataset = dataset.WPlusGeneratingDataset()
  dataloader = DataLoader(imgTrainDataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=0,
                          pin_memory=True)
  optimizer = optim.AdamW(encoder.parameters())

  num_epochs = 30
  feat_lambda = 1.0

  def intHandler(sigNum, frame):
    weight_name = 'encoder_weights_pretrain_test.pth'
    save_weights = input("\nsave weights? y/[n]:")
    if save_weights == 'y':
      torch.save(encoder.state_dict(), weight_name)
      print(f"saved weights to {weight_name}")
    exit(0)
  signal(SIGINT, intHandler)

  def compute_loss(wps, imgs):
      wps = wps.to('cuda')
      imgs = imgs.to('cuda')
      enc_wps = encoder(imgs).view(-1, 14, 512)
      loss = F.mse_loss(wps, enc_wps, 
                        reduction='mean')
      return loss

  def train_loop():
    losshist = np.zeros(len(dataloader))
    for i, data in enumerate(tqdm(dataloader)):
      wps, imgs = data
      optimizer.zero_grad()
      loss = compute_loss(wps, imgs)
      loss.backward()
      optimizer.step()
      losshist[i] = loss.cpu().detach().numpy()
    return np.mean(losshist)

  for epoch in range(15):
    loss = train_loop()
    print(f'{epoch}: loss={loss}')

  optimizer = optim.SGD(encoder.parameters(),
                        lr=0.0001, momentum=0.9)
  scheduler = optim.lr_scheduler.StepLR(optimizer, 
                                        step_size=15, 
                                        gamma=0.1)

  for epoch in range(num_epochs):
    loss = train_loop()
    print(f'{epoch+15}: loss={loss}')
    scheduler.step()

  torch.save(encoder.state_dict(), 'encoder_weights_pretrain2.pth')


if __name__ == "__main__":
  main()
