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


def canny_plus_coherence_transport(img_bgr: np.ndarray) -> np.ndarray:
    """ 
    Remove pelos usando detecção de bordas Canny + inpainting por transporte de coerência.
    Algoritmo mais avançado que pode preservar melhor texturas e bordas da lesão, mas é mais complexo

    Args:
        img_bgr: Imagem no espaço BGR (uint8).

    Returns:
        Imagem sem pelos no espaço BGR (uint8).
    
    """
    gray_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Convert to float for Wiener filter
    gray_float = gray_bgr.astype(np.float64)

    # Filter the image to reduce noise (ignorando os avisos irritantes do console)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        filtered_float = wiener(gray_float, (3, 3))

    # Remove NaN values that may arise from the Wiener filter
    filtered_float = np.nan_to_num(filtered_float, nan=0.0)
    filtered_uint8 = np.clip(filtered_float, 0, 255).astype(np.uint8)

    # Calculate the median intensity of the image
    median = np.median(filtered_uint8)

    lower = int(max(0, 0.7 * median))
    upper = int(min(255, 1.3 * median))

    # Apply Canny detection
    edges = cv2.Canny(filtered_uint8, lower, upper)

    # Dilate the edges to create a thicker mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)

    # Inpaint the original image using the dilated edge mask    
    inpainted = cv2.inpaint(img_bgr, dilated_edges, 3, cv2.INPAINT_NS)

    return inpainted

def laplacian_of_gaussian(img_bgr: np.ndarray) -> np.ndarray:
    """
    Remove pelos usando o filtro Laplacian of Gaussian (LoG) para detectar bordas finas, seguido de inpainting.
    Algoritmo é simples e pode ser eficaz para casos onde os pelos são finos e contrastantes, mas pode não ser tão robusto quanto os métodos baseados em morfologia ou gradientes direcionais.

    Args:
        img_bgr: Imagem no espaço BGR (uint8).

    Returns:
        Imagem sem pelos no espaço BGR (uint8).
    """
    # Convert to grayscale
    gray_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian of Gaussian
    log = gaussian_laplace(gray_bgr, sigma=1)

    # Threshold the LoG result to create a binary mask
    _, mask = cv2.threshold(log.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)

    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)

    # Inpaint the original image using the mask
    inpainted = cv2.inpaint(img_bgr, closed_mask, 3, cv2.INPAINT_TELEA)

    return inpainted


def no_removal(img_bgr: np.ndarray) -> np.ndarray:
    """Passthrough — sem remoção de pelos (para experimentos de ablação)."""
    return img_bgr


# Mapa para seleção dinâmica via config.yaml
METHODS = {
    "sharp_razor": sharp_razor,
    "dull_razor": dull_razor,
    "canny_plus_coherence_transport": canny_plus_coherence_transport,
    "laplacian_of_gaussian": laplacian_of_gaussian,
    "none": no_removal
}
