import clip
import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as T
from text_train import FallbackEncoder
from generator import Generator
from text_textenc import TextEncoder
from utils.visualizer import save_image


def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser()
  parser.add_argument('--mode', type=str,
                      default='man', help='Mode (gen for generation, man for manipulation).')
  parser.add_argument('--description', type=str, default='he is old',
                      help='The description.')
  parser.add_argument('--image_path', type=str, default='examples/142.jpg', help='Path of images to invert.')
  return parser.parse_args()

def mix_styles(wc, ws):
  wc = wc.view(14, 512)
  ws = ws.view(14, 512)
  wp = np.zeros((14,512))
  wp[0] = wc.to('cpu').detach().numpy()[0]
  wp[1] = wc.to('cpu').detach().numpy()[1]
  for i in range(2, 14):
    wp[i] = ws.to('cpu').detach().numpy()[i]

  return torch.Tensor(wp).to('gpu')
  

def main():
  args = parse_args()
  generator = Generator()
  imgenc = FallbackEncoder(True)
  textenc = TextEncoder()
  textenc.load_state_dict(torch.load('text_encoder_weights.pth'))
  textenc.to('cuda')
  textenc.eval()

  if args.mode == 'man':
    img = Image.open(args.image_path)
    img = T.functional.reize(img, (256, 256))
    img = T.functional.to_tensor(img)
    img = torch.unsqueeze(img, 0)
    img_wp = imgenc(img)
    text_wp = textenc(clip.tokenize(args.description))
    wc = text_wp

  elif args.mode == 'gen':
    text_wp = textenc(clip.tokenize(args.description))
    samp_wp = generator.G.sample(1, latent_space_type='wp',
                           z_space_dim=512, num_layers=14)
    samp_wp = generator.G.preprocess(samp_wp, latent_space_type='wp')
    img_wp = torch.Tensor(samp_wp)
    wc = text_wp
    ws = img_wp

  else:
    raise ValueError("not a valid mode")


  wp = mix_styles(wc, ws)
  reimg = generator.generate_timg_from_wp(wp)
  reimg = generator.G.postprocess(reimg.cpu().detach().numpy())[0]
  save_image('out_img.jpg', reimg)

if __name__ == "__main__":
  main()