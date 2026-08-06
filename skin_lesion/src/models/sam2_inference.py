"""
SAM 2 (Segment Anything Model 2) — interface para segmentação de lesões.

O SAM 2 é um Foundation Model da Meta que segmenta qualquer objeto a partir
de um "prompt" visual. Diferente da U-Net (que aprende a segmentar lesões
especificamente), o SAM 2 é zero-shot: nunca foi treinado no ISIC 2018,
mas consegue segmentar a partir de uma bounding box ou ponto.

Estratégia usada aqui: bounding box centrada na imagem com margem de 10%.
Essa é a mesma lógica do GrabCut — assume que a lesão está centralizada,
o que é válido para o ISIC 2018.

Instalação:
    O SAM 2 precisa ser instalado do repositório oficial antes de usar este módulo:
    git clone https://github.com/facebookresearch/sam2.git
    cd sam2
    pip install -e .

    Os checkpoints ficam em sam2/checkpoints/ (baixar com download_ckpts.sh ou wget).
    Para RTX 3050 (4GB), use sam2.1_hiera_tiny ou sam2.1_hiera_small.
"""

import numpy as np
import cv2
import torch


class SAM2Segmenter:
    """
    Interface entre o pipeline do projeto e o SAM 2.

    Args:
        model_cfg:       Nome do arquivo de config do SAM 2.
                         Deve bater com o checkpoint baixado.
                         Opções: 'sam2.1_hiera_tiny.yaml', 'sam2.1_hiera_small.yaml',
                                 'sam2.1_hiera_base_plus.yaml', 'sam2.1_hiera_large.yaml'
        checkpoint_path: Caminho absoluto para o arquivo .pt baixado.
        margin_ratio:    Margem da bounding box automática como fração da imagem.
                         0.10 = 10% de cada lado (mesmo padrão do GrabCut).
    """

    def __init__(
        self,
        model_cfg: str = "sam2.1_hiera_tiny.yaml",
        checkpoint_path: str = "",
        margin_ratio: float = 0.10,
    ):
        # Import local — só falha se o SAM 2 não estiver instalado,
        # não quebra o resto do projeto na importação
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError:
            raise ImportError(
                "SAM 2 não encontrado. Instale com:\n"
                "  git clone https://github.com/facebookresearch/sam2.git\n"
                "  cd sam2 && pip install -e ."
            )

        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.margin_ratio  = margin_ratio
        print(f"Carregando SAM 2 ({model_cfg}) em: {self.device}")

        sam2_model       = build_sam2(model_cfg, checkpoint_path, device=self.device)
        self.predictor   = SAM2ImagePredictor(sam2_model)

        # bfloat16 autocast recomendado pela Meta para GPUs Ampere+ (RTX 30xx em diante)
        if self.device.type == "cuda":
            self._autocast = torch.autocast("cuda", dtype=torch.bfloat16)
            self._autocast.__enter__()

        print("SAM 2 carregado.")

    def segment(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Segmenta a lesão usando uma bounding box centrada como prompt.

        Args:
            img_bgr: Imagem pré-processada no espaço BGR (uint8).
                     Deve ter passado pelo pipeline completo (color constancy,
                     hair removal, CLAHE, resize) antes de chegar aqui.

        Returns:
            Máscara binária uint8 {0, 255} com shape (H, W).
        """
        # SAM 2 trabalha com RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(img_rgb)

        # Bounding box automática com margem — mesma lógica do GrabCut
        h, w = img_bgr.shape[:2]
        mx   = int(w * self.margin_ratio)
        my   = int(h * self.margin_ratio)
        box  = np.array([mx, my, w - mx, h - my])  # [x_min, y_min, x_max, y_max]

        masks, scores, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=box[None, :],      # batch dimension esperada pelo SAM 2
            multimask_output=False, # retorna só a máscara de maior score
        )

        # masks shape: (1, H, W) bool → converte para uint8 {0, 255}
        return (masks[0].astype(np.uint8)) * 255
