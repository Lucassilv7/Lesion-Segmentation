"""
Utilitários compartilhados entre módulos e notebooks.
"""

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """
    Carrega uma imagem BGR do disco.

    Args:
        path: Caminho para o arquivo de imagem.

    Returns:
        Imagem no formato BGR (uint8).

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o arquivo não puder ser lido como imagem.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {path}")

    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {path}")

    return img


def show_comparison(images: dict[str, np.ndarray], figsize: tuple = (15, 5)) -> None:
    """
    Exibe múltiplas imagens lado a lado para comparação visual.
    Útil nos notebooks — não deve ser chamado em scripts de processamento em lote.

    Args:
        images:  Dicionário {título: imagem_bgr}.
        figsize: Tamanho da figura matplotlib.
    """
    import matplotlib.pyplot as plt

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (title, img) in zip(axes, images.items()):
        if img.ndim == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
