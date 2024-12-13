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
from tqdm import tqdm
from types import SimpleNamespace
from typing import Dict, List, Optional, Union

from . import extractors
from .utils.base_model import dynamic_load
from .utils.io import list_h5_names, read_image
from .utils.parsers import parse_image_lists
from .image_dataset import ImgDataset
from .config import configs

# # configs for different models
# configs = {
#     "superpoint": {
#         "out_dir": "superpoint-feats-n4096-r1024",
#         "model_stuff": {
#             "name": "superpoint",
#             "nms_radius": 3,
#             "max_kp": 4096,
#         },
#         "preprocess": {
#             "gray": True,
#             "max_size": 1024,
#         },
#     },
#     "sift": {
#         "output": "feats-sift",
#         "model": {"name": "dog"},
#         "preprocessing": {
#             "grayscale": True,
#             "resize_max": 1600,
#         },
#     },
# }

@torch.no_grad()
def extract_features(
    cfg: Dict,
    img_dir: Path,
    out_dir: Optional[Path] = None,
    half_precision: bool = True,
    img_list: Optional[Union[Path, List[str]]] = None,
    feat_path: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:

    data = ImgDataset(img_dir, cfg["preprocess"], img_list)
    if not feat_path:
        feat_path = Path(out_dir, cfg["out_dir"] + ".h5")
    feat_path.parent.mkdir(exist_ok=True, parents=True)

    done_imgs = set(list_h5_names(feat_path) if feat_path.exists() and not overwrite else ())
    data.imgs = [n for n in data.imgs if n not in done_imgs]
    if not data.imgs:
        return feat_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = dynamic_load(extractors, "superpoint_wrapper")
    model = Model(cfg["model_stuff"]).eval().to(device)

    loader = torch.utils.data.DataLoader(data, num_workers=1, shuffle=False, pin_memory=True)

    for idx, batch in enumerate(tqdm(loader)):
        name = data.imgs[idx]
        out = model({"image": batch["img"].to(device, non_blocking=True)})
        out = {k: v[0].cpu().numpy() for k, v in out.items()}

        out["image_size"] = orig_size = batch["orig_size"][0].numpy()
        if "keypoints" in out:
            size = np.array(batch["img"].shape[-2:][::-1])
            scales = (orig_size/size).astype(np.float32)
            out["keypoints"] = (out["keypoints"] + 0.5) * scales[None] - 0.5
            if "scales" in out:
                out["scales"] *= scales.mean()
            uncert = getattr(model, "detection_noise", 1) * scales.mean()

        if half_precision:
            for k in out:
                if out[k].dtype == np.float32:
                    out[k] = out[k].astype(np.float16)

        with h5py.File(str(feat_path), "a", libver="latest") as f:
            try:
                if name in f:
                    del f[name]
                g = f.create_group(name)
                for k, v in out.items():
                    g.create_dataset(k, data=v)
                if "keypoints" in out:
                    g["keypoints"].attrs["uncertainty"] = uncert
            except OSError as e:
                if "No space left on device" in e.args[0]:
                    del g, f[name]
                raise e

        del out
    return feat_path

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

# class ImgDataset(torch.utils.data.Dataset):
#     defaults = {
#         "patterns": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
#         "gray": False,
#         "max_size": None,
#         "force_resize": False,
#         "resize_method": "cv2_area",
#     }

#     def __init__(self, root, cfg, img_list=None):
#         self.cfg = SimpleNamespace(**{**self.defaults, **cfg})
#         self.root = root

#         if not img_list:
#             paths = []
#             for pat in self.cfg.patterns:
#                 paths += glob.glob((Path(root)/"**"/pat).as_posix(), recursive=True)
#             if not paths:
#                 raise ValueError(f"No images found in {root}")
#             paths = sorted(set(paths))
#             self.imgs = [Path(p).relative_to(root).as_posix() for p in paths]
#             # logger.info(f"Found {len(self.imgs)} images in {root}")
#         else:
#             if isinstance(img_list, (Path, str)):
#                 self.imgs = parse_image_lists(img_list)
#             elif isinstance(img_list, colls.Iterable):
#                 self.imgs = [p.as_posix() if isinstance(p, Path) else p for p in img_list]
#             else:
#                 raise ValueError(f"Bad img_list format: {img_list}")

#             for img in self.imgs:
#                 if not (root/img).exists():
#                     raise ValueError(f"Image {img} not found in {root}")

#     def __getitem__(self, idx):
#         img = read_image(self.root/self.imgs[idx], self.cfg.gray)
#         img = img.astype(np.float32)
#         size = img.shape[:2][::-1]

#         if self.cfg.max_size and (self.cfg.force_resize or max(size) > self.cfg.max_size):
#             scale = self.cfg.max_size / max(size)
#             new_size = tuple(int(round(x*scale)) for x in size)
#             img = resize_img(img, new_size, self.cfg.resize_method)

#         if self.cfg.gray:
#             img = img[None]
#         else:
#             img = img.transpose((2,0,1))
#         img = img/255.0

#         return {
#             "img": img,
#             "orig_size": np.array(size),
#         }

#     def __len__(self):
#         return len(self.imgs)



# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--img_dir", type=Path, required=True)
#     parser.add_argument("--out_dir", type=Path, required=True)
#     parser.add_argument("--cfg", type=str, default="superpoint", choices=list(configs.keys()))
#     parser.add_argument("--half", action="store_true")
#     parser.add_argument("--img_list", type=Path)
#     parser.add_argument("--feat_path", type=Path)
#     args = parser.parse_args()
#     extract_features(
#         configs[args.cfg],
#         args.img_dir,
#         args.out_dir,
#         args.half,
#         args.img_list,
#         args.feat_path,
#     )
