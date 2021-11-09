import clip
import torch.nn as nn

class TextEncoder(nn.Module):
  def __init__(self):
    super().__init__()

    self.text_enc = clip.load('RN50')
    self.fc = nn.Linear(1024, 512*14)
    self.activate = nn.LeakyReLU(neagtive_slope=0.2, inplace=True)

  def forward(self, x):
    clipfeat = self.text_enc.encode_text(x)
    return self.activate(self.fc(clipfeat))