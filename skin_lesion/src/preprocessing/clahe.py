"""
CLAHE — Contrast Limited Adaptive Histogram Equalization.

Aplicado no canal L do espaço LAB para realçar bordas da lesão sem
distorcer as cores reais da imagem.
"""

import cv2
import numpy as np


def apply_clahe(
    img_bgr: np.ndarray,
    clip_limit: float = 0.5,
    tile_grid_size: tuple = (8, 8),
) -> np.ndarray:
    """
    Aplica CLAHE no canal de luminosidade (LAB) de uma imagem BGR.

    Args:
        img_bgr:        Imagem no espaço BGR (uint8).
        clip_limit:     Limite de contraste (valores menores = resultado mais suave).
        tile_grid_size: Tamanho da grade de tiles para o histograma local.

    Returns:
        Imagem realçada no espaço BGR (uint8).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe_obj.apply(l_channel)

    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
