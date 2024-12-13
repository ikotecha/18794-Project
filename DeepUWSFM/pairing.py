import argparse
from pathlib import Path
from typing import List, Optional, Union
import collections.abc as colls

from .utils.io import list_h5_names
from .utils.parsers import parse_image_lists

def main(
    out_file: Path,
    imgs: Optional[Union[Path, List[str]]] = None,
):
    # Get query image names
    if imgs:
        if isinstance(imgs, (str, Path)):
            q_names = parse_image_lists(imgs)
        elif isinstance(imgs, colls.Iterable):
            q_names = list(imgs)
        else:
            raise ValueError(f"Bad image list type: {imgs}")
    elif feats:
        q_names = list_h5_names(feats)
    else:
        raise ValueError("Need either images or features!")

    match_self = True
    ref_names = q_names

    # Make all pairs
    pairs = []
    for i, img1 in enumerate(q_names):
        for j, img2 in enumerate(ref_names):
            if match_self and j <= i:
                continue
            pairs.append((img1, img2))

    # Save pairs to file
    with open(out_file, "w") as f:
        f.write("\n".join(f"{i} {j}" for i, j in pairs))
