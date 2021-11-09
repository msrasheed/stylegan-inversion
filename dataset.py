import pickle
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

BASE_DIR = '/home/mrasheed/595CV/'
CELEB_DIR = os.path.join(BASE_DIR, "CelebAMask-HQ")
TEDI_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_FILENAMES_FILE = os.path.join(TEDI_DIR, 'train/filenames.pickle')
IMG_DIR = os.path.join(CELEB_DIR, "CelebA-HQ-img")

class CelebAHQImgDataset(Dataset):
  def __init__(self):
    with open(TRAIN_FILENAMES_FILE, 'rb') as trainpkl:
      self.files = pickle.load(trainpkl)
    self.transform = T.Compose([
      T.Resize((256, 256)),
      T.ToTensor()
    ])
    
  def __len__(self):
    return len(self.files)
  
  def __getitem__(self, idx):
    file = self.files[idx]
    filename = os.path.join(IMG_DIR, file + '.jpg')
    image = Image.open(filename)
    return self.transform(image)
