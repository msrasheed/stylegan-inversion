import os
import torch
import dataset
import encoder_net
from generator import Generator
from utils.visualizer import save_image
from tqdm import tqdm
import project
import numpy as np
from encoder import Encoder

def extract_img_from_stylegan_tensor(x):
  images = x.cpu().detach().numpy()
  images = (images - (-1)) * 255 / (1 - (-1))
  images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
  images = images.transpose(0, 2, 3, 1)
  return images

def main():
  # data = dataset.testImgDataset()
  data = dataset.testImgDataset()
  generator = project.loadStyleGan()
  encoder = Encoder(use_fallback=False,
                    weights_path='encoder_weights_pretrain2.pth')

  for idx in tqdm(range(len(data))):
    img = data[idx]
    img = torch.unsqueeze(img, 0).to('cuda')
    wp = encoder(img).view(-1, 14, 512)
    reimg = generator(wp)
    reimg = extract_img_from_stylegan_tensor(reimg)[0]
    save_image(os.path.join('gen_imgs', data.files[idx] + '.jpg'), reimg)


if __name__ == "__main__":
  main()
#layer  inch    outch   dim
# 0       3       64    256
# 1      64      128    128
# 2     128      256     64
# 3     256      512     32
# 4     512     1024     16
# 5    1024     1024      8
# 6    1024     1024      4
# 7 1024*16   512*14
