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

"""
This file uses/adapts functionality from the hloc (Hierarchical-Localization) project
Repository: https://github.com/cvg/Hierarchical-Localization
"""

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