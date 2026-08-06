from .unet import UNet
from .dataset import SkinLesionDataset

# SAM2Segmenter importado sob demanda — requer instalação separada do sam2
# Use: from src.models.sam2_inference import SAM2Segmenter
try:
    from .sam2_inference import SAM2Segmenter
    __all__ = ["UNet", "SkinLesionDataset", "SAM2Segmenter"]
except ImportError:
    __all__ = ["UNet", "SkinLesionDataset"]
