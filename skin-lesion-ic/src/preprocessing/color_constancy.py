"""
Algoritmos de constância de cor para normalização de iluminação.

Objetivo: reequilibrar a iluminação das imagens dermatoscópicas para que
todas aparentem ter sido capturadas sob uma luz branca neutra, reduzindo
a variação de cor entre imagens do dataset.

Referência: Buchsbaum (1980) — Gray World; Finlayson & Trezzi (2004) — Shades of Gray.
"""

import cv2
import numpy as np


def gray_world(img_bgr: np.ndarray) -> np.ndarray:
    """
    Assume que a média de todas as cores na imagem deve ser cinza neutro.
    Equivale ao Shades of Gray com p=1.

    Args:
        img_bgr: Imagem no espaço BGR (uint8).

    Returns:
        Imagem corrigida no espaço BGR (uint8).
    """
    img_b, img_g, img_r = cv2.split(img_bgr)

    avg_r = np.mean(img_r)
    avg_g = np.mean(img_g)
    avg_b = np.mean(img_b)
    avg_all = (avg_r + avg_g + avg_b) / 3

    scale_r = np.clip(img_r * (avg_all / avg_r), 0, 255).astype(np.uint8)
    scale_g = np.clip(img_g * (avg_all / avg_g), 0, 255).astype(np.uint8)
    scale_b = np.clip(img_b * (avg_all / avg_b), 0, 255).astype(np.uint8)

    return cv2.merge([scale_b, scale_g, scale_r])


def shades_of_gray(img_bgr: np.ndarray, p: int = 6) -> np.ndarray:
    """
    Generalização do Gray World usando a norma Minkowski de ordem p.
    Com p=1 equivale ao Gray World; p grande aproxima o Max-RGB.
    p=6 é o valor padrão recomendado na literatura para imagens de pele.

    Args:
        img_bgr: Imagem no espaço BGR (uint8).
        p:       Ordem da norma Minkowski.

    Returns:
        Imagem corrigida no espaço BGR (uint8).
    """
    img_b, img_g, img_r = cv2.split(img_bgr)

    img_r_p = np.power(img_r.astype(np.float32), p)
    img_g_p = np.power(img_g.astype(np.float32), p)
    img_b_p = np.power(img_b.astype(np.float32), p)

    root_r = np.power(np.mean(img_r_p), 1 / p)
    root_g = np.power(np.mean(img_g_p), 1 / p)
    root_b = np.power(np.mean(img_b_p), 1 / p)

    max_ilum = max(root_r, root_g, root_b)

    ilum_r = np.clip(img_r.astype(np.float32) * (max_ilum / root_r), 0, 255).astype(np.uint8)
    ilum_g = np.clip(img_g.astype(np.float32) * (max_ilum / root_g), 0, 255).astype(np.uint8)
    ilum_b = np.clip(img_b.astype(np.float32) * (max_ilum / root_b), 0, 255).astype(np.uint8)

    return cv2.merge([ilum_b, ilum_g, ilum_r])


# Mapa para seleção dinâmica via config.yaml
METHODS = {
    "shades_of_gray": shades_of_gray,
    "gray_world": gray_world,
}
