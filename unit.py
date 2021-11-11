import unittest
import torch
import numpy as np
from utils.visualizer import load_image
import torchvision.transforms as T
import encoder_net as en
import dataset as data
import generator as gen
from models.perceptual_model import PerceptualModel


@unittest.skip('not now')
class TestEncoderNet(unittest.TestCase):
  def setUp(self):
    self.enc = en.EncoderNet()
    self.dataset = data.CelebAHQImgDataset()

  def test_init(self):
    self.assertIsInstance(self.enc, en.EncoderNet)

  def test_shape_output(self):
    x = self.dataset[10]
    x = torch.unsqueeze(x, 0)
    x = self.enc(x)
    self.assertTrue(x.shape == (1, 512 * 14))


@unittest.skip('not now')
class TestImgDataset(unittest.TestCase):
  def setUp(self):
    self.dataset = data.CelebAHQImgDataset()

  def test_len(self):
    self.assertTrue(len(self.dataset) == 24000)

  def test_load(self):
    image = self.dataset[10]
    self.assertIsInstance(image, torch.Tensor)
    self.assertTrue(image.shape == (3, 256, 256))


@unittest.skip("takes too long and is working right now")
class TestGenerator(unittest.TestCase):
  def setUp(self):
    self.generator = gen.Generator()

  def check_image_shape(self, image):
    self.assertTrue(image[0].shape == (256, 256, 3))

  def test_generate(self):
    image = self.generator.generate_img()
    self.check_image_shape(image)

  def test_generate_from_wp(self):
    wp = torch.normal(torch.zeros(512*14) + .5)
    image = self.generator.generate_timg_from_wp(wp)
    self.assertIsInstance(image, torch.Tensor)
    self.assertTrue(image.shape[1:] == (3, 256, 256))

  def test_generate_from_wps(self):
    wp = torch.normal(torch.zeros(4*512*14) + .5)
    images = self.generator.generate_timgs_from_wps(wp)
    self.assertIsInstance(images, torch.Tensor)
    self.assertTrue(images.shape[1:] == (3, 256, 256))


@unittest.skip('not now')
class TestPerceptualModel(unittest.TestCase):
  def setUp(self):
    self.feat_model = PerceptualModel(-1, 1)

  def test_get_img_feats(self):
    test_imgs = torch.zeros(4 * 256 * 256 * 3).view(4, 3, 256, 256)
    feats = self.feat_model.net(test_imgs)
    self.assertTrue(feats.shape[0] == 4)


class TestPretrainDataset(unittest.TestCase):
  def setUp(self):
    self.data = data.WPlusGeneratingDataset()

  def test_gen_sample(self):
    wp, img = self.data[0]
    self.assertIsInstance(wp, torch.Tensor)
    self.assertTrue(wp.shape == (14, 512),
        msg=f"wp shape = {wp.shape}")
    self.assertIsInstance(img, torch.Tensor)
    self.assertTrue(img.shape == (3, 256, 256),
        msg=f"img shape = {img.shape}")
    
    
if __name__ == "__main__":
  unittest.main()
