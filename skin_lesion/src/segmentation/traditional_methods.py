import cv2
import numpy as np
from collections import deque

def segment_otsu(img_bgr: np.ndarray) -> np.ndarray:
    """
    Segmentação utilizando o método de Limiarização de Otsu.
    Ideal para lesões com alto contraste em relação à pele.
    """
    # 1. Converte para escala de cinza
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 2. Aplica Otsu. Usamos THRESH_BINARY_INV porque a lesão geralmente 
    # é mais escura que a pele, e queremos a máscara da lesão como branca (255).
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    return mask

def segment_watershed(img_bgr: np.ndarray) -> np.ndarray:
    """
    Segmentação utilizando Watershed.
    Usa Otsu como base, mas aplica mapa de distâncias para lidar 
    melhor com bordas irregulares e texturas.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 1. Limpeza de ruído (Morfologia)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 2. Encontrar a área que com certeza é fundo (pele)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 3. Encontrar a área que com certeza é a lesão (centro)
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # 4. Encontrar região desconhecida (as bordas difusas)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 5. Criar marcadores
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1 # Fundo vira 1 (não 0)
    markers[unknown == 255] = 0 # Região desconhecida vira 0

    # 6. Aplicar Watershed (modifica os marcadores na própria variável)
    img_copy = img_bgr.copy()
    cv2.watershed(img_copy, markers)

    # 7. Criar máscara final (Onde marcador for maior que 1, é o nosso objeto/lesão)
    mask = np.zeros_like(gray, dtype=np.uint8)
    mask[markers > 1] = 255
    
    return mask

def segment_grabcut(img_bgr: np.ndarray, rect: tuple = None, iterations: int = 5) -> np.ndarray:
    """
    Segmentação utilizando GrabCut.
    Baseado em grafos, precisa de um "palpite" (bounding box) de onde está a lesão.
    """
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    bgd_model = np.zeros((1,65), np.float64)
    fgd_model = np.zeros((1,65), np.float64)

    # Se nenhum retângulo foi passado, assumimos que a lesão está centralizada
    # Criamos um retângulo que corta as bordas da imagem (margem de 10%)
    if rect is None:
        h, w = img_bgr.shape[:2]
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.1)
        # Formato do rect: (x, y, largura, altura)
        rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)

    # Aplica o algoritmo
    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

    # A máscara do GrabCut retorna 4 valores:
    # 0 = Background certo, 2 = Background provável
    # 1 = Foreground certo, 3 = Foreground provável
    # Transformamos 1 e 3 em branco (255), e o resto em preto (0)
    binary_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')
    
    return binary_mask

_PALETA = [
    (230,  25,  75), ( 60, 180,  75), (255, 225,  25), (  0, 130, 200),
    (245, 130,  48), (145,  30, 180), ( 70, 240, 240), (240,  50, 230),
    (210, 245,  60), (250, 190, 212), (  0, 128, 128), (220, 190, 255),
    (170, 110,  40), (255, 250, 200), (128,   0,   0), (170, 255, 195),
    (128, 128,   0), (255, 215, 180), (  0,   0, 128), (128, 128, 128),
]

_VIZINHOS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def segment_region_growing(img: np.ndarray, seeds: list, T: float) -> np.ndarray:
    """
    Segmentação por crescimento de região com BFS.
 
    Parâmetros:
        img     : imagem RGB de entrada
        sementes: lista de tuplas (y, x)  — coordenadas das sementes
        T       : limiar de diferença de intensidade
 
    Retorna imagem pseudocolorida RGB onde cada região tem uma cor distinta
    e pixels não atribuídos ficam pretos.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    labels = np.zeros((height, width), dtype=np.int32)  # 0 = sem rótulo

    queeue = deque()

    # Inicializa sementes
    for idx, (sy, sx) in enumerate(seeds):
        label = idx + 1  
        labels[sy, sx] = label
        queeue.append((sy, sx, label, gray[sy, sx]))

    # BFS
    while queeue:
        y, x, label, seed_value = queeue.popleft()
        for dy, dx in _VIZINHOS_8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width and labels[ny, nx] == 0:
                if abs(float(gray[ny, nx]) - float(seed_value)) <= T:
                    labels[ny, nx] = label
                    queeue.append((ny, nx, label, seed_value))
    
    # Pseudocolorização
    results = np.zeros((height, width, 3), dtype=np.uint8)
    for label in range(1, len(seeds) + 1):
        color = _PALETA[(label - 1) % len(_PALETA)]
        mask = labels == label
        results[mask] = color
    
    return results



METHODS = {
    "otsu": segment_otsu,
    "watershed": segment_watershed,
    "grabcut": segment_grabcut,
    "region_growing": segment_region_growing,
}