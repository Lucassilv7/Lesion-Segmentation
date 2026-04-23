"""
Algoritmos de remoção digital de pelos em imagens dermatoscópicas.

Pelos são um dos artefatos mais comuns e problemáticos: podem obscurecer
a lesão, criar bordas falsas e confundir modelos de segmentação.

Algoritmos disponíveis
----------------------
- sharp_razor : baseado em gradientes direcionais + filtragem geométrica (recomendado)
- dull_razor  : baseado em morfologia (black-hat) + inpainting — clássico da literatura
"""

import warnings

import cv2
import numpy as np
from scipy.ndimage import gaussian_laplace
from scipy.signal import wiener
from skimage.filters import threshold_li
from skimage.measure import label, regionprops


def sharp_razor(
    img_bgr: np.ndarray,
    min_area: int = 10,
    max_solidity: float = 0.9,
    inpaint_radius: int = 3,
) -> np.ndarray:
    """
    Remove pelos usando gradientes direcionais no canal vermelho + filtragem
    geométrica por propriedades de região (área e solidez).

    Pelos são estruturas alongadas (baixa solidez) e pequenas. A filtragem
    geométrica descarta objetos que não têm essas características, evitando
    remover partes da própria lesão.

    Args:
        img_bgr:       Imagem no espaço BGR (uint8).
        min_area:      Área mínima (px²) para um componente ser considerado pelo.
        max_solidity:  Solidez máxima. Objetos mais sólidos (circulares) são ignorados.
        inpaint_radius: Raio do inpainting (cv2.INPAINT_NS).

    Returns:
        Imagem sem pelos no espaço BGR (uint8).
    """
    red_channel = img_bgr[:, :, 2]

    # Kernels de gradiente direcionais (0°, 45°, -45°)
    k_0 = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], dtype=np.float32)
    k_45 = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]], dtype=np.float32)
    k_m45 = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]], dtype=np.float32)

    response = (
        np.abs(cv2.filter2D(red_channel, cv2.CV_32F, k_0))
        + np.abs(cv2.filter2D(red_channel, cv2.CV_32F, k_45))
        + np.abs(cv2.filter2D(red_channel, cv2.CV_32F, k_m45))
    )
    response = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Limiarização por entropia (Li)
    threshold = threshold_li(response)
    brute_mask = (response > threshold).astype(np.uint8) * 255

    # Filtragem geométrica: mantém apenas regiões com perfil de pelo
    labeled = label(brute_mask)
    clean_mask = np.zeros_like(brute_mask)
    for prop in regionprops(labeled):
        if prop.area >= min_area and prop.solidity <= max_solidity:
            for r, c in prop.coords:
                clean_mask[r, c] = 255

    # Dilata levemente para cobrir bordas dos pelos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)

    return cv2.inpaint(img_bgr, clean_mask, inpaint_radius, cv2.INPAINT_NS)


def dull_razor(img_bgr: np.ndarray, inpaint_radius: int = 6) -> np.ndarray:
    """
    Remove pelos usando morfologia black-hat + inpainting TELEA.
    Algoritmo clássico da literatura (Lee et al., 1997).

    Args:
        img_bgr:       Imagem no espaço BGR (uint8).
        inpaint_radius: Raio do inpainting.

    Returns:
        Imagem sem pelos no espaço BGR (uint8).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    inpainted = cv2.inpaint(img_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)
    return cv2.medianBlur(inpainted, 3)


def no_removal(img_bgr: np.ndarray) -> np.ndarray:
    """Passthrough — sem remoção de pelos (para experimentos de ablação)."""
    return img_bgr


# Mapa para seleção dinâmica via config.yaml
METHODS = {
    "sharp_razor": sharp_razor,
    "dull_razor": dull_razor,
    "none": no_removal,
}
