"""
Utilitários compartilhados entre módulos e notebooks.

Divididos em dois grupos:
- Utilitários de imagem (load_image, show_comparison) — usados nos notebooks de pré-processamento e segmentação
- Utilitários de treino (get_loaders, save/load_checkpoint, check_accuracy, save_predictions_as_imgs) — usados no notebook de treino
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision


# =============================================================================
# Utilitários de imagem
# =============================================================================

def load_image(path: str | Path) -> np.ndarray:
    """
    Carrega uma imagem BGR do disco.

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


# =============================================================================
# Utilitários de treino
# =============================================================================

def save_checkpoint(state: dict, filepath: str | Path) -> None:
    """
    Salva estado do modelo e otimizador em disco.

    Args:
        state:    Dicionário com 'epoch', 'state_dict', 'optimizer', 'val_dice'.
        filepath: Caminho do arquivo .pth a salvar.
    """
    print(f"=> Salvando checkpoint (epoch {state.get('epoch', '?')}, "
          f"Dice={state.get('val_dice', 0):.4f})...")
    torch.save(state, filepath)


def load_checkpoint(
    filepath: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict:
    """
    Carrega estado do modelo (e opcionalmente do otimizador) de um checkpoint.

    Args:
        filepath:  Caminho do arquivo .pth.
        model:     Instância do modelo que receberá os pesos.
        optimizer: Instância do otimizador (opcional — passe None para inferência).

    Returns:
        O dicionário completo do checkpoint (para recuperar epoch, val_dice etc.).
    """
    print(f"=> Carregando checkpoint: {filepath}")
    state = torch.load(filepath, map_location="cpu")
    model.load_state_dict(state["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state


def get_loaders(
    train_dir: str,
    train_maskdir: str,
    val_dir: str,
    val_maskdir: str,
    batch_size: int,
    train_transform,
    val_transform,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple:
    """
    Cria e retorna os DataLoaders de treino e validação.

    Returns:
        (train_loader, val_loader)
    """
    # Import local para evitar dependência circular no topo do arquivo
    from src.models.dataset import SkinLesionDataset
    from torch.utils.data import DataLoader

    train_ds = SkinLesionDataset(train_dir, train_maskdir, transform=train_transform)
    val_ds   = SkinLesionDataset(val_dir,   val_maskdir,   transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


def check_accuracy(
    loader,
    model: torch.nn.Module,
    device: str = "cuda",
) -> dict:
    """
    Calcula pixel accuracy e Dice no loader fornecido.

    Args:
        loader: DataLoader de validação ou teste.
        model:  Modelo a avaliar.
        device: 'cuda' ou 'cpu'.

    Returns:
        Dicionário com 'accuracy', 'dice'.
    """
    num_correct = 0
    num_pixels  = 0
    dice_sum    = 0.0  # bug corrigido: variável não estava sendo inicializada

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            preds = (torch.sigmoid(model(x)) > 0.5).float()
            num_correct += (preds == y).sum().item()
            num_pixels  += torch.numel(preds)
            dice_sum    += (2 * (preds * y).sum() / ((preds + y).sum() + 1e-8)).item()

    accuracy = num_correct / num_pixels
    dice     = dice_sum / len(loader)

    print(f"Acurácia : {num_correct}/{num_pixels} ({accuracy:.4f})")
    print(f"Dice     : {dice:.4f}")
    model.train()

    return {"accuracy": accuracy, "dice": dice}


def save_predictions_as_imgs(
    loader,
    model: torch.nn.Module,
    folder: str = "../results/saved_images/",
    device: str = "cuda",
) -> None:
    """
    Salva as predições do modelo e as máscaras de ground truth como PNGs,
    para inspeção visual após o treino.

    Gera pares: pred_{idx}.png e true_{idx}.png para cada batch.

    Args:
        loader: DataLoader (normalmente de validação).
        model:  Modelo treinado.
        folder: Pasta de destino.
        device: 'cuda' ou 'cpu'.
    """
    model.eval()
    Path(folder).mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            x     = x.to(device)
            preds = (torch.sigmoid(model(x)) > 0.5).float()

            torchvision.utils.save_image(preds,            f"{folder}/pred_{idx}.png")
            torchvision.utils.save_image(y.unsqueeze(1),   f"{folder}/true_{idx}.png")

    model.train()
