import os
import torch
import dataset
import encoder_net
from generator import Generator
from utils.visualizer import save_image
from tqdm import tqdm


def main():
  data = dataset.testImgDataset()
  generator = Generator()
  encoder = encoder_net.EncoderNet()
  encoder.load_state_dict(torch.load('encoder_weights.pth'))
  encoder.to('cuda')
  encoder.eval()

  for idx in tqdm(range(len(data))):
    img = data[idx]
    img = torch.unsqueeze(img, 0).to('cuda')
    wp = encoder(img)
    reimg = generator.generate_timg_from_wp(wp)
    reimg = generator.G.postprocess(reimg.cpu().detach().numpy())[0]
    save_image(os.path.join('gen_imgs2', data.files[idx] + '.jpg'), reimg)


if __name__ == "__main__":
  main()