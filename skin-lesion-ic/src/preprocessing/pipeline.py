"""
Pipeline de pré-processamento de imagens dermatoscópicas.

Ordem das etapas (definida em config.yaml):
    1. Color Constancy  — normaliza iluminação entre imagens
    2. Hair Removal     — remove pelos (artefato mais comum)
    3. CLAHE            — realça contraste da lesão
    4. Resize           — padroniza resolução para a rede
    5. Normalize        — escala pixels para [0, 1]

Uso rápido
----------
    from src.preprocessing.pipeline import Pipeline
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    pipeline = Pipeline(cfg)
    img_processed = pipeline.run(img_bgr)

    # Para processar um dataset inteiro:
    pipeline.run_dataset("data/raw/images", "results/preprocessed_imgs")
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from .clahe import apply_clahe
from .color_constancy import METHODS as CC_METHODS
from .hair_removal import METHODS as HR_METHODS

# Mapa de interpolação para cv2.resize
_INTERPOLATION = {
    "bilinear": cv2.INTER_LINEAR,
    "nearest_neighbor": cv2.INTER_NEAREST,
    "bicubic": cv2.INTER_CUBIC,
}


class Pipeline:
    """
    Pipeline configurável de pré-processamento.

    Todos os hiperparâmetros são lidos de config.yaml, nunca hardcoded aqui.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["preprocessing"]

    # ------------------------------------------------------------------
    # Etapas individuais (podem ser chamadas isoladamente nos notebooks)
    # ------------------------------------------------------------------

    def step_color_constancy(self, img: np.ndarray) -> np.ndarray:
        c = self.cfg["color_constancy"]
        fn = CC_METHODS[c["method"]]
        if c["method"] == "shades_of_gray":
            return fn(img, p=c["p"])
        return fn(img)

    def step_hair_removal(self, img: np.ndarray) -> np.ndarray:
        c = self.cfg["hair_removal"]
        fn = HR_METHODS[c["method"]]
        if c["method"] == "sharp_razor":
            return fn(
                img,
                min_area=c["min_area"],
                max_solidity=c["max_solidity"],
                inpaint_radius=c["inpaint_radius"],
            )
        if c["method"] == "dull_razor":
            return fn(img, inpaint_radius=c["inpaint_radius"])
        return fn(img)

    def step_clahe(self, img: np.ndarray) -> np.ndarray:
        c = self.cfg["clahe"]
        return apply_clahe(
            img,
            clip_limit=c["clip_limit"],
            tile_grid_size=tuple(c["tile_grid_size"]),
        )

    def step_resize(self, img: np.ndarray, is_mask: bool = False) -> np.ndarray:
        c = self.cfg["resize"]
        size = (c["width"], c["height"])
        interp_key = c["mask_interpolation"] if is_mask else c["image_interpolation"]
        return cv2.resize(img, size, interpolation=_INTERPOLATION[interp_key])

    def step_normalize(self, img: np.ndarray) -> np.ndarray:
        if not self.cfg["normalize"]["enabled"]:
            return img
        return (img.astype(np.float32) / self.cfg["normalize"]["scale"])

    # ------------------------------------------------------------------
    # Execução completa
    # ------------------------------------------------------------------

    def run(self, img_bgr: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Aplica todas as etapas em sequência em uma única imagem.

        Args:
            img_bgr:   Imagem BGR uint8.
            normalize: Se False, retorna uint8 (útil para salvar/visualizar).

        Returns:
            Imagem processada (float32 [0,1] se normalize=True, uint8 caso contrário).
        """
        img = self.step_color_constancy(img_bgr)
        img = self.step_hair_removal(img)
        img = self.step_clahe(img)
        img = self.step_resize(img)
        if normalize:
            img = self.step_normalize(img)
        return img

    def run_dataset(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        extensions: tuple = (".jpg", ".jpeg", ".png"),
    ) -> None:
        """
        Processa todas as imagens de uma pasta e salva os resultados.

        Args:
            input_dir:  Pasta com as imagens originais.
            output_dir: Pasta de destino das imagens processadas.
            extensions: Extensões de arquivo a processar.
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_paths = sorted(
            p for p in input_dir.iterdir() if p.suffix.lower() in extensions
        )

        if not image_paths:
            print(f"[Pipeline] Nenhuma imagem encontrada em: {input_dir}")
            return

        print(f"[Pipeline] Processando {len(image_paths)} imagens...")
        for i, path in enumerate(image_paths, 1):
            img = cv2.imread(str(path))
            if img is None:
                print(f"  [!] Não foi possível carregar: {path.name}")
                continue

            processed = self.run(img, normalize=False)  # salva como uint8
            out_path = output_dir / path.name
            cv2.imwrite(str(out_path), processed)

            if i % 50 == 0 or i == len(image_paths):
                print(f"  {i}/{len(image_paths)} concluídas")

        print(f"[Pipeline] Imagens salvas em: {output_dir}")


# ------------------------------------------------------------------
# Utilitário: carrega config e instancia o pipeline em uma linha
# ------------------------------------------------------------------

def load_pipeline(config_path: str | Path = "config.yaml") -> Pipeline:
    """
    Carrega o config.yaml e retorna um Pipeline pronto para uso.

    Exemplo:
        pipeline = load_pipeline()
        result = pipeline.run(img_bgr)
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return Pipeline(cfg)
