import argparse
import contextlib
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pycolmap
from tqdm import tqdm

from .utils.database import COLMAPDatabase
from .utils.geometry import compute_epipolar_errors
from .utils.io import get_keypoints, get_matches
from .utils.parsers import parse_retrieval

"""
This file uses/adapts functionality from the hloc (Hierarchical-Localization) project
Repository: https://github.com/cvg/Hierarchical-Localization
"""

def import_features(
    image_ids: Dict[str, int], database_path: Path, features_path: Path
):
    db = COLMAPDatabase.connect(database_path)

    for image_name, image_id in tqdm(image_ids.items()):
        keypoints = get_keypoints(features_path, image_name)
        keypoints += 0.5  # COLMAP origin
        db.add_keypoints(image_id, keypoints)

    db.commit()
    db.close()


def import_matches(
    image_ids: Dict[str, int],
    database_path: Path,
    pairs_path: Path,
    matches_path: Path,
    min_match_score: Optional[float] = None,
):
    with open(str(pairs_path), "r") as f:
        pairs = [p.split() for p in f.readlines()]

    db = COLMAPDatabase.connect(database_path)

    matched = set()
    for name0, name1 in tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
        matches, scores = get_matches(matches_path, name0, name1)
        if min_match_score:
            matches = matches[scores > min_match_score]
        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        db.add_two_view_geometry(id0, id1, matches)

    db.commit()
    db.close()

def run_triangulation(
    model_path: Path,
    database_path: Path,
    image_dir: Path,
    reference_model: pycolmap.Reconstruction,
    verbose: bool = False,
    options: Optional[Dict[str, Any]] = None,
) -> pycolmap.Reconstruction:
    model_path.mkdir(parents=True, exist_ok=True)
    if options is None:
        options = {}
    with pycolmap.ostream():
        reconstruction = pycolmap.triangulate_points(
            reference_model, database_path, image_dir, model_path, options=options
        )
    return reconstruction