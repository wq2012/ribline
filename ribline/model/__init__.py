"""Model components for 3D rib centerline detection, topology matching, and refinement."""

from ribline.model.anatomy import (
    RibCageStatisticalAtlas,
    extract_thoracic_frame,
    get_anatomical_rib_mask,
    get_polar_angle,
    standardize_hu,
)
from ribline.model.features import FeatureExtractor
from ribline.model.forest import RibForest
from ribline.model.metrics import (
    aggregate_subject_metrics,
    centerline_dice,
    chamfer_distance,
    dilated_dice,
    evaluate_subject_ribs,
    hausdorff_distance_95,
)
from ribline.model.reconstruction import CenterlineExtractor
from ribline.model.refinement import DeformableRibRefiner, RibSnake
from ribline.model.topology import BilateralViterbiLabeler, RibTopologyMatcher

try:
    from ribline.model.unet.model import RibUNet, RibUNet3D
except ImportError:  # pragma: no cover - optional when running CPU-only SCF without torch
    RibUNet = None
    RibUNet3D = None

__all__ = [
    "RibCageStatisticalAtlas",
    "extract_thoracic_frame",
    "get_anatomical_rib_mask",
    "get_polar_angle",
    "standardize_hu",
    "FeatureExtractor",
    "RibForest",
    "RibUNet",
    "RibUNet3D",
    "CenterlineExtractor",
    "RibTopologyMatcher",
    "BilateralViterbiLabeler",
    "RibSnake",
    "DeformableRibRefiner",
    "chamfer_distance",
    "hausdorff_distance_95",
    "centerline_dice",
    "dilated_dice",
    "evaluate_subject_ribs",
    "aggregate_subject_metrics",
]
