import os
import clip
import pickle
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

BASE_DIR = '/home/mrasheed/595CV/'
CELEB_DIR = os.path.join(BASE_DIR, "CelebAMask-HQ")
TEDI_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_FILENAMES_FILE = os.path.join(TEDI_DIR, 'train/filenames.pickle')
TEST_FILENAMES_FILE = os.path.join(TEDI_DIR, 'test/filenames.pickle')
IMG_DIR = os.path.join(CELEB_DIR, "CelebA-HQ-img")
TEXT_DIR = os.path.join(TEDI_DIR, 'text/celeba-caption')

def trainImgDataset():
  return CelebAHQImgDataset2(TRAIN_FILENAMES_FILE, IMG_DIR)
def testImgDataset():
  return CelebAHQImgDataset2(TEST_FILENAMES_FILE, IMG_DIR)
def trainTextDataset():
  return CelebAHQTextDataset()


class CelebAHQImgDataset2(Dataset):
  def __init__(self, files_file, data_dir):
    with open(files_file, 'rb') as trainpkl:
      self.files = pickle.load(trainpkl)
    self.data_dir = data_dir
    self.transform = T.Compose([
      T.Resize((256, 256)),
      T.ToTensor()
    ])
    
  def __len__(self):
    # return len(self.files)
    return 3000
  
  def __getitem__(self, idx):
    file = self.files[idx]
    filename = os.path.join(self.data_dir, file + '.jpg')
    image = Image.open(filename)
    return self.transform(image)
    
class CelebAHQTextDataset(Dataset):
  def __init__(self):
    self.imgdata = trainImgDataset()

  def __len__(self):
    return len(self.imgdata)
  
  def __getitem__(self, idx):
    filename = self.imgdata.files[idx]
    filename = os.path.join(TEXT_DIR, filename + '.txt')
    with open(filename, 'r') as descfile:
      desc = descfile.read().split('\n')
    desc_tokens = clip.tokenize(desc)    
    return desc_tokens, self.imgdata[idx]
