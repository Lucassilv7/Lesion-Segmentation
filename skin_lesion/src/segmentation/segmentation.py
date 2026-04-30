
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from .traditional_methods import METHODS as TRAD_METHODS


class Segmentation:
   
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["segmentation"]
    
    def step_segment_traditional_otsu(self, img: np.ndarray) -> np.ndarray:
        #c = self.cfg["traditional_method"]
        fn = TRAD_METHODS["otsu"]
        return fn(img)
    
    def step_segment_traditional_watershed(self, img: np.ndarray) -> np.ndarray:
        #c = self.cfg["traditional_method"]
        fn = TRAD_METHODS["watershed"]
        return fn(img)
    
    def step_segment_traditional_grabcut(self, img: np.ndarray) -> np.ndarray:
        #c = self.cfg["traditional_method"]
        fn = TRAD_METHODS["grabcut"]
        return fn(img)
    
    # ------------------------------------------------------------------
    # Execução completa
    # ------------------------------------------------------------------

    def run(self, img: np.ndarray) -> np.ndarray:
        """
        Executa o pipeline completo de segmentação, retornando a máscara binária.
        """

        mask = self.step_segment_traditional_grabcut(img)
        return mask
    
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
            print(f"[Segmentation] Nenhuma imagem encontrada em: {input_dir}")
            return
        
        print(f"[Segmentation] Processando {len(image_paths)} imagens...")
        for i, path in enumerate(image_paths, 1):
            img = cv2.imread(str(path))
            mask = self.run(img)
            output_path = output_dir / path.name
            cv2.imwrite(str(output_path), mask)

            if i % 50 == 0 or i == len(image_paths):
                print(f"  {i}/{len(image_paths)} concluídas")
        
        print(f"[Segmentation] Máscaras salvas em: {output_dir}")

# ------------------------------------------------------------------
# Utilitário: carrega config e instancia o segmentation em uma linha
# ------------------------------------------------------------------

def load_segmentation(config_path: str | Path = "config.yaml") -> Segmentation:
    """
    Carrega o config.yaml e retorna um Segmentation pronto para uso.

    Exemplo:
        segmentation = load_segmentation()
        mask = segmentation.run(img_bgr)
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    return Segmentation(cfg)