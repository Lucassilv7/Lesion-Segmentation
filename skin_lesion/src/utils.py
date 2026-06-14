"""
Utilitários compartilhados entre módulos e notebooks.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from models.dataset import SkinLesionDataset
from torch.utils.data import DataLoader


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

def save_checkpoint(state: dict, filename: str) -> None:
    """
    Salva o estado do modelo e otimizador em um arquivo.

    Args:
        state: Dicionário contendo os estados do modelo e otimizador.
        filename: Caminho para o arquivo de checkpoint.
    """
    print("=> Salvando checkpoint...")
    torch.save(state, filename)

def load_checkpoint(checkpoint: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
    """
    Carrega o estado do modelo e otimizador de um arquivo de checkpoint.

    Args:
        checkpoint: Caminho para o arquivo de checkpoint.
        model: Instância do modelo a ser carregado.
        optimizer: Instância do otimizador a ser carregado.
    """
    print("=> Carregando checkpoint...")
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    optimizer.load_state_dict(state["optimizer"])

def get_loaders(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        batch_size,
        train_transform,
        val_transform,
        num_workers=4,
        pin_memory=True,
) -> tuple[DataLoader, DataLoader]:
    """
    Cria DataLoaders para os conjuntos de treinamento e validação.

    Args:
        train_dir: Diretório das imagens de treinamento.
        train_maskdir: Diretório das máscaras de treinamento.
        val_dir: Diretório das imagens de validação.
        val_maskdir: Diretório das máscaras de validação.
        batch_size: Tamanho do lote.
        train_transform: Transformações a serem aplicadas nas imagens de treinamento.
        val_transform: Transformações a serem aplicadas nas imagens de validação.
        num_workers: Número de subprocessos para carregar os dados.
        pin_memory: Se True, os tensores serão alocados na memória fixa.

    Returns:
        Tuple contendo os DataLoaders de treinamento e validação.
    """
    train_ds = SkinLesionDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    val_ds = SkinLesionDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader: DataLoader, model: torch.nn.Module, device: torch.device = "cuda") -> None:
    """
    Avalia a acurácia do modelo em um DataLoader.

    Args:
        loader: DataLoader contendo os dados de avaliação.
        model: Modelo a ser avaliado.
        device: Dispositivo (CPU ou GPU) para realizar a avaliação.
    """
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8 )

    print(f"Acurácia: {num_correct}/{num_pixels} ({num_correct/num_pixels:.4f})")
    print(f"Score de Dice: {dice_score/len(loader):.4f}")
    model.train()

def save_predictions_as_imgs(loader: DataLoader, model: torch.nn.Module, folder: str = "../results/saved_images/", device: torch.device = "cuda") -> None:
    """
    Salva as previsões do modelo como imagens PNG.

    Args:
        loader: DataLoader contendo os dados de avaliação.
        model: Modelo a ser avaliado.
        folder: Diretório onde as imagens serão salvas.
        device: Dispositivo (CPU ou GPU) para realizar a avaliação.
    """
    model.eval()
    Path(folder).mkdir(exist_ok=True)

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x = x.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
            torchvision.utils.save_image(y.unsqueeze(1), f"{folder}/true_{idx}.png")

    model.train()