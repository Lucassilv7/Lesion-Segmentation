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
from sklearn.metrics import jaccard_score
import SimpleITK as sitk 

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

def calculate_jaccard_dice(
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> dict[str, float]:
    """
    Calcula o Índice de Jaccard (IoU) e o Coeficiente de Dice e ntre a máscara verdadeira e a predita.

    Args:
        true_mask: Máscara binária verdadeira (0 ou 1).
        pred_mask: Máscara binária predita (0 ou 1).

    Returns:
        Dicionário com os valores do Índice de Jaccard e do Coeficiente de Dice.
    """
    true_flat = true_mask.flatten()
    pred_flat = pred_mask.flatten()
    return {
        "jaccard": jaccard_score(true_flat.astype(np.uint8), pred_flat.astype(np.uint8)),
        "dice": calculate_dice_coefficient(true_mask, pred_mask)
    }

def calculate_dice_coefficient(
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> float:
    """
    Calcula o Coeficiente de Dice entre a máscara verdadeira e a predita.

    Args:
        true_mask: Máscara binária verdadeira (0 ou 1).
        pred_mask: Máscara binária predita (0 ou 1).

    Returns:
        Valor do Coeficiente de Dice (entre 0 e 1).
    """
    dice = sitk.LabelOverlapMeasuresImageFilter()
    dice.Execute(sitk.GetImageFromArray(true_mask.astype(np.uint8)),
                 sitk.GetImageFromArray(pred_mask.astype(np.uint8)))
    return dice.GetDiceCoefficient()

def evaluate_segmentation(
    true_mask: np.ndarray,
    variantes: dict[str, np.ndarray],
) -> list[dict]:
    """
    Avalia múltiplas variantes de segmentação em relação à máscara verdadeira.

    Args:
        true_mask: Máscara binária verdadeira (0 ou 1).
        variantes: Dicionário {nome_variante: máscara_predita}.

    Returns:
        Lista de dicionários com 'jaccard' e 'dice'.
    """
    results = []
    for name, pred_mask in variantes.items():
        metrics = calculate_jaccard_dice(true_mask, pred_mask)
        results.append({"name": name, **metrics})
    return results