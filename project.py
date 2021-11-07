import os
from inverter import Inverter
from utils.visualizer import save_image, load_image, resize_image

def main():
  inverter = Inverter()
  viz_results = inverter.generate_img()

  save_image(next_img_name(), viz_results[0])

def next_img_name(path='.'):
  imgfiles = [imgfile for imgfile in os.listdir(path) 
              if imgfile.split('.')[0].isdecimal()]
  maxnum = int(max(imgfiles).split('.')[0]) if len(imgfiles) else 0
  return str(maxnum + 1) + '.png'


if __name__ == "__main__":
  main()