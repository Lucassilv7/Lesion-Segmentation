from .segmentation import Segmentation

from .traditional_methods import segment_otsu, segment_watershed, segment_grabcut

__all__ = [
    "segment_otsu",
    "segment_watershed",
    "segment_grabcut",
    "Segmentation"
 
]
