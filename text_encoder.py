import clip
import torch
import torch.nn as nn

class TextEncoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.text_enc = clip.load('RN50')
    self.fc = nn.Linear(1024, 512*14)
    self.activate = nn.LeakyReLU(negative_slope=0.2, inplace=True)

  def forward(self, x):
    clipfeat = self.text_enc[0].encode_text(x.view(-1, 77))
    return self.activate(self.fc(clipfeat.view(-1, 11, 1024).type(torch.float32)))
