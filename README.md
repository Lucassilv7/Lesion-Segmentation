# Segmentação Automática de Lesões Dermatológicas — IC UFERSA

Iniciação Científica focada em segmentação de lesões de pele usando redes neurais convolucionais.
Dataset: [ISIC 2018](https://challenge.isic-archive.com/data/#2018).

## Estrutura do Projeto

```
skin-lesion-ic/
├── config.yaml                  ← todos os hiperparâmetros aqui
├── requirements.txt
│
├── data/
│   ├── raw/                     ← imagens originais do ISIC (não versionar)
│   └── processed/               ← splits de treino/val/test
│
├── src/
│   ├── preprocessing/
│   │   ├── pipeline.py          ← orquestrador principal
│   │   ├── color_constancy.py
│   │   ├── hair_removal.py
│   │   └── clahe.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── utils.py
│
├── notebooks/
│   ├── 01_explore_dataset.ipynb
│   ├── 02_preprocessing.ipynb   ← comparação de métodos e parâmetros
│   └── 03_eval_methods.ipynb
│
└── results/
    ├── figures/                 ← gráficos salvos
    ├── metrics/                 ← CSVs de avaliação
    └── preprocessed_imgs/       ← imagens processadas
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

## Uso Rápido

```python
from src.preprocessing import load_pipeline
from src.utils import load_image

pipeline = load_pipeline("config.yaml")
img = load_image("data/raw/ISIC_0000074.jpg")
result = pipeline.run(img)          # retorna float32 [0, 1]
result_uint8 = pipeline.run(img, normalize=False)  # retorna uint8 para visualizar
```

Para explorar e comparar métodos, abra o notebook:
```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

## Pipeline de Pré-processamento

| Etapa | Método | Motivo |
|-------|--------|--------|
| Color Constancy | Shades of Gray (p=6) | Normaliza iluminação entre imagens do dataset |
| Hair Removal | Sharp Razor | Remove pelos sem afetar bordas da lesão |
| CLAHE | clip_limit=0.5 no canal L (LAB) | Realça contraste da lesão sem distorcer cores |
| Resize | 256×256 bilinear | Padroniza resolução para a U-Net |
| Normalize | ÷ 255 | Escala para [0, 1] esperado pela rede |

Configurações detalhadas em `config.yaml`.
