"""Hugging Face Hub helpers for downloading and loading pretrained RibLine models."""

import json
import os
from typing import Dict, List, Optional

from ribline.model.anatomy import RibCageStatisticalAtlas
from ribline.model.features import FeatureExtractor
from ribline.model.forest import RibForest
from ribline.model.refinement import RibSnake
from ribline.model.topology import RibTopologyMatcher

DEFAULT_REPO_ID = "wq2012/ribline-3d-ct-centerline"

MODEL_REPOS: Dict[str, str] = {
    "full": "wq2012/ribline-3d-ct-centerline",
    "anatomical-scf": "wq2012/ribline-anatomical-scf",
    "unet": "wq2012/ribline-3d-unet",
    "pass2": "wq2012/ribline-scf-pass2-autocontext",
}

REPO_DEFAULT_FILES: Dict[str, List[str]] = {
    "wq2012/ribline-3d-ct-centerline": [
        "rib_atlas.joblib",
        "rib_forest_anatomical.joblib",
        "feature_config_anatomical.json",
        "rib_unet_3d.pth",
        "train_config_unet.json",
    ],
    "wq2012/ribline-anatomical-scf": [
        "rib_atlas.joblib",
        "rib_forest_anatomical.joblib",
        "feature_config_anatomical.json",
    ],
    "wq2012/ribline-3d-unet": [
        "rib_atlas.joblib",
        "rib_unet_3d.pth",
        "train_config_unet.json",
    ],
    "wq2012/ribline-scf-pass2-autocontext": [
        "rib_atlas.joblib",
        "rib_forest_pass1.joblib",
        "rib_forest_pass2.joblib",
        "feature_config_pass1.json",
        "feature_config_img.json",
        "feature_config_ctx.json",
    ],
}


def download_pretrained_models(
    repo_id: str = DEFAULT_REPO_ID,
    local_dir: Optional[str] = None,
    filenames: Optional[List[str]] = None,
    token: Optional[str] = None,
) -> Dict[str, str]:
    """Downloads pretrained RibLine checkpoints from Hugging Face Hub.

    If ``local_dir`` already contains a requested file locally, that local path
    is returned directly without making a network request.

    Args:
        repo_id: Hugging Face repository ID or short alias (``"full"``,
            ``"anatomical-scf"``, ``"unet"``, ``"pass2"``).
        local_dir: Optional local directory to store or read downloaded files.
        filenames: Optional explicit list of filenames to fetch.
        token: Optional Hugging Face token (defaults to cached user token).

    Returns:
        Dictionary mapping each filename to its resolved local file path.
    """
    resolved_repo = MODEL_REPOS.get(repo_id, repo_id)
    if filenames is None:
        filenames = list(
            REPO_DEFAULT_FILES.get(
                resolved_repo, REPO_DEFAULT_FILES[DEFAULT_REPO_ID]
            )
        )

    resolved_paths: Dict[str, str] = {}
    missing_files: List[str] = []
    for fname in filenames:
        if local_dir is not None:
            cand = os.path.join(local_dir, fname)
            if os.path.exists(cand):
                resolved_paths[fname] = os.path.abspath(cand)
                continue
        missing_files.append(fname)

    if missing_files:
        from huggingface_hub import hf_hub_download

        for fname in missing_files:
            dl_kwargs = {"repo_id": resolved_repo, "filename": fname}
            if local_dir is not None:
                dl_kwargs["local_dir"] = local_dir
            if token is not None:
                dl_kwargs["token"] = token
            resolved_paths[fname] = hf_hub_download(**dl_kwargs)

    return resolved_paths


def load_pretrained_models(
    repo_id: str = DEFAULT_REPO_ID,
    local_dir: Optional[str] = None,
    device: str = "cpu",
    load_unet: bool = True,
    load_scf: bool = True,
    token: Optional[str] = None,
) -> dict:
    """Downloads and instantiates pretrained RibLine models from Hugging Face Hub.

    Args:
        repo_id: Hugging Face repository ID or short alias.
        local_dir: Optional local directory containing or caching checkpoints.
        device: Torch device for the 3D U-Net (default: ``"cpu"``).
        load_unet: Whether to load the 3D U-Net when present in the repository.
        load_scf: Whether to load the Semantic Context Forest when present.
        token: Optional Hugging Face token.

    Returns:
        Dictionary containing instantiated components (``atlas``, ``matcher``,
        ``snake``, and as applicable ``scf_anatomical``, ``feature_extractor``,
        ``unet``, ``scf_pass1``, ``scf_pass2``, and ``paths``).
    """
    resolved_repo = MODEL_REPOS.get(repo_id, repo_id)
    default_files = list(
        REPO_DEFAULT_FILES.get(resolved_repo, REPO_DEFAULT_FILES[DEFAULT_REPO_ID])
    )
    if not load_unet:
        default_files = [
            f for f in default_files if f not in ("rib_unet_3d.pth", "train_config_unet.json")
        ]
    if not load_scf:
        default_files = [
            f
            for f in default_files
            if not f.startswith("rib_forest_") and not f.startswith("feature_config_")
        ]

    paths = download_pretrained_models(
        repo_id=resolved_repo,
        local_dir=local_dir,
        filenames=default_files,
        token=token,
    )

    bundle: dict = {"repo_id": resolved_repo, "local_dir": local_dir, "paths": paths}
    if "rib_atlas.joblib" in paths:
        atlas = RibCageStatisticalAtlas.load(paths["rib_atlas.joblib"])
        bundle["atlas"] = atlas
        bundle["matcher"] = RibTopologyMatcher(atlas=atlas)
    else:
        bundle["atlas"] = None
        bundle["matcher"] = RibTopologyMatcher()
    bundle["snake"] = RibSnake()

    if load_scf and "rib_forest_anatomical.joblib" in paths:
        bundle["scf_anatomical"] = RibForest.load(
            paths["rib_forest_anatomical.joblib"]
        )
        bundle["forest"] = bundle["scf_anatomical"]
        if "feature_config_anatomical.json" in paths:
            with open(paths["feature_config_anatomical.json"], "r") as f:
                bundle["feature_extractor"] = FeatureExtractor.from_config(
                    json.load(f)
                )

    if load_scf and "rib_forest_pass1.joblib" in paths:
        bundle["scf_pass1"] = RibForest.load(paths["rib_forest_pass1.joblib"])
    if load_scf and "rib_forest_pass2.joblib" in paths:
        bundle["scf_pass2"] = RibForest.load(paths["rib_forest_pass2.joblib"])

    if load_unet and "rib_unet_3d.pth" in paths:
        from ribline.model.unet.model import RibUNet3D

        bundle["unet"] = RibUNet3D.load(
            paths["rib_unet_3d.pth"], map_location=device
        )

    return bundle
