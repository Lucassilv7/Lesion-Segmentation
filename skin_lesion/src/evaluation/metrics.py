"""
Métricas de avaliação do pipeline de pré-processamento.

SSIM e PSNR medem similaridade estrutural e de sinal em relação à imagem
original. São usadas aqui para comparar o impacto de diferentes configurações
do pipeline — não para avaliar a segmentação final (que usará Dice/IoU).

Nota: SSIM e PSNR têm limitações para avaliar pré-processamento de imagens
médicas. Uma imagem com SSIM baixo pode ser, na prática, melhor para a
segmentação por ter realçado estruturas relevantes. Use as métricas como
guia exploratório, não como verdade absoluta.
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim



def calculate_ssim_psnr(
    original: np.ndarray,
    processed: np.ndarray,
) -> dict[str, float]:
    """
    Calcula SSIM e PSNR entre a imagem original e a processada.

    Args:
        original:  Imagem original BGR (uint8).
        processed: Imagem processada BGR (uint8).

    Returns:
        Dicionário com as chaves 'ssim' e 'psnr'.
    """
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    return {
        "ssim": ssim(orig_gray, proc_gray),
        "psnr": psnr(orig_gray, proc_gray),
    }


def evaluate_pipeline_variants(
    img_bgr: np.ndarray,
    variants: dict[str, np.ndarray],
) -> list[dict]:
    """
    Avalia múltiplas variantes do pipeline em relação à imagem original.

    Args:
        img_bgr:  Imagem original BGR.
        variants: Dicionário {nome_variante: imagem_processada}.

    Returns:
        Lista de dicionários com 'name', 'ssim', 'psnr'.
    """
    results = []
    for name, processed in variants.items():
        metrics = calculate_ssim_psnr(img_bgr, processed)
        results.append({"name": name, **metrics})
    return results
