import argparse
import collections.abc as colls
import glob
import pprint
from pathlib import Path
import cv2
import h5py
import numpy as np
import PIL.Image
import torch
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

from .utils.io import read_image
from .utils.parsers import parse_image_lists

def resize_img(img, size, method):
    if method.startswith('cv2_'):
        m = getattr(cv2, 'INTER_' + method[4:].upper())
        h, w = img.shape[:2]
        if m == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            m = cv2.INTER_LINEAR
        new_img = cv2.resize(img, size, interpolation=m)
    elif method.startswith('pil_'):
        m = getattr(PIL.Image, method[4:].upper())
        new_img = PIL.Image.fromarray(img.astype(np.uint8))
        new_img = new_img.resize(size, resample=m)
        new_img = np.asarray(new_img, dtype=img.dtype)
    else:
        raise ValueError(f"Bad resize method: {method}")
    return new_img

class ImgDataset(torch.utils.data.Dataset):
  defaults = {
      "patterns": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
      "gray": False,
      "max_size": None,
      "force_resize": False,
      "resize_method": "cv2_area",
  }

  def __init__(self, root, cfg, img_list=None):
      self.cfg = SimpleNamespace(**{**self.defaults, **cfg})
      self.root = root

      if not img_list:
          paths = []
          for pat in self.cfg.patterns:
              paths += glob.glob((Path(root)/"**"/pat).as_posix(), recursive=True)
          if not paths:
              raise ValueError(f"No images found in {root}")
          paths = sorted(set(paths))
          self.imgs = [Path(p).relative_to(root).as_posix() for p in paths]
      else:
          if isinstance(img_list, (Path, str)):
              self.imgs = parse_image_lists(img_list)
          elif isinstance(img_list, colls.Iterable):
              self.imgs = [p.as_posix() if isinstance(p, Path) else p for p in img_list]
          else:
              raise ValueError(f"Bad img_list format: {img_list}")

          for img in self.imgs:
              if not (root/img).exists():
                  raise ValueError(f"Image {img} not found in {root}")

  def __getitem__(self, idx):
      img = read_image(self.root/self.imgs[idx], self.cfg.gray)
      img = img.astype(np.float32)
      size = img.shape[:2][::-1]

      if self.cfg.max_size and (self.cfg.force_resize or max(size) > self.cfg.max_size):
          scale = self.cfg.max_size / max(size)
          new_size = tuple(int(round(x*scale)) for x in size)
          img = resize_img(img, new_size, self.cfg.resize_method)

      if self.cfg.gray:
          img = img[None]
      else:
          img = img.transpose((2,0,1))
      img = img/255.0

      return {
          "img": img,
          "orig_size": np.array(size),
      }

  def __len__(self):
      return len(self.imgs)