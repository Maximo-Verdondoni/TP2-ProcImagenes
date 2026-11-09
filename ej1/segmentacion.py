# =============================================================================
# segmentacion.py - MÓDULO COMPARTIDO DEL PROBLEMA 1
# -----------------------------------------------------------------------------
# Reúne la lógica de los pasos 3 y 4 para que paso4.py, paso5.py (y el script
# final) la reutilicen sin copiarla:
#   segmentar()         -> Punto A, etapa 1: máscara de objetos y componentes (paso 3)
#   refinar_objetos()   -> Punto A, etapa 2: clasifica monedas/dados por forma y
#                          refina la máscara de cada uno (paso 4)
# Todo trabaja sobre la imagen reducida a 1/4 (968x670), donde las monedas miden
# 64-92 px de diámetro.
# =============================================================================
import cv2
import numpy as np

RUTA_IMG = "ej1/monedas.jpg"
ESCALA = 4


def cargar_imagen(ruta=RUTA_IMG, f=ESCALA):
    """Devuelve (imagen original, imagen reducida a 1/f), ambas en BGR."""
    img = cv2.imread(ruta)
    assert img is not None, f"No se encontró {ruta}"
    # INTER_AREA promedia los píxeles al reducir (evita aliasing, Unidad 2.1)
    small = cv2.resize(img, None, fx=1 / f, fy=1 / f, interpolation=cv2.INTER_AREA)
    return img, small


def _fondo_canal(c, k=101, sigma=15):
    """Estima el fondo de UN canal: mediana grande + suavizado (resultado del paso 1)."""
    # Se reduce a la mitad (1/8 del original) por velocidad: el fondo es suave
    tiny = cv2.resize(c, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    # Mediana 101x101: a esta escala las monedas miden 32-46 px y ocupan una
    # fracción chica de la ventana, así que la mediana devuelve el valor del fondo
    f = cv2.medianBlur(tiny, k)
    f = cv2.GaussianBlur(f, (0, 0), sigma)    # suaviza los escalones de la mediana
    return cv2.resize(f, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_LINEAR)


def _rellenar_huecos(m):
    """Rellena huecos internos: se inunda el fondo desde la esquina (0,0) y lo que
    no queda inundado es hueco. Supone que (0,0) es fondo (de ahí el assert)."""
    assert m[0, 0] == 0, "La esquina (0,0) no es fondo"
    ff = m.copy()
    pad = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)   # floodFill exige una máscara 2 px más grande
    cv2.floodFill(ff, pad, (0, 0), 255)
    return m | cv2.bitwise_not(ff)    # not(ff) = solo los huecos; el OR los rellena


def segmentar(small, t_alto=25, t_bajo=14, n_grow=10, area_min=1000):
    """PASO 3 (Punto A, etapa 1): distancia Lab al fondo, histéresis y morfología.
    Devuelve un diccionario con las imágenes intermedias y los componentes conectados."""
    # CIELAB (Unidad 5.4): separa luminancia (L) de color (a*, b*). Necesario porque
    # las monedas doradas tienen luminancia parecida al fondo pero difieren en b*.
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    fondo_lab = np.dstack([_fondo_canal(lab[:, :, i]) for i in range(3)])

    # Distancia euclídea al fondo en Lab: un solo valor por píxel que combina
    # diferencia de luminancia y de color (en float32: en uint8 la resta desborda)
    diff = lab.astype(np.float32) - fondo_lab.astype(np.float32)
    dist = np.sqrt((diff ** 2).sum(axis=2))
    dist = cv2.GaussianBlur(dist, (0, 0), 2)     # atenúa ruido y grabado de las monedas

    # Umbral con histéresis mediante dilatación geodésica de n_grow pasos (Unidades 3.1 y 6.5;
    # iterar hasta que no cambie sería la reconstrucción completa, que acá se evita a propósito):
    #  - semillas (dist > 25): píxeles que seguro son objeto (valle del histograma)
    #  - zona candidata (dist > 14): permite completar bordes débiles
    #  - crecimiento: se dilata la semilla n_grow veces SOLO dentro de la zona candidata.
    #    Acotarlo a 10 iteraciones (máx. 10 px) evita que el halo de sombra una
    #    objetos vecinos.
    seeds = ((dist > t_alto) * 255).astype(np.uint8)
    low = ((dist > t_bajo) * 255).astype(np.uint8)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask0 = seeds.copy()
    for _ in range(n_grow):
        mask0 = cv2.dilate(mask0, k3) & low

    # Morfología (Unidad 6):
    #  clausura 7x7: sella fisuras; relleno de huecos: máscaras sólidas (relieves, puntos);
    #  apertura 31x31: elimina restos y corta puentes finos entre objetos pegados
    #  (seguro, porque las monedas miden 64-92 px, bastante más que el elemento)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_close = cv2.morphologyEx(mask0, cv2.MORPH_CLOSE, k_close)
    mask_fill = _rellenar_huecos(mask_close)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask_open = cv2.morphologyEx(mask_fill, cv2.MORPH_OPEN, k_open)

    # Componentes conectados (Unidad 3.3); el 0 es el fondo. area_min descarta ruido
    # (la moneda más chica tiene ~3200 px de área a esta escala).
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask_open, connectivity=8)
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= area_min]
    return dict(lab=lab, dist=dist, mask0=mask0, mask_open=mask_open,
                labels=labels, stats=stats, cents=cents, keep=keep)


def _contorno_mayor(mask):
    """Contorno externo de mayor área de una máscara binaria."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(cnts, key=cv2.contourArea)


def _rellenar_sin_esquina(m):
    """Dibuja el contorno externo mayor relleno: tapa todos los huecos internos
    (los puntos del dado) sin depender de la esquina (0,0). A diferencia de
    _rellenar_huecos, también sirve cuando el objeto está recortado."""
    out = np.zeros_like(m)
    cv2.drawContours(out, [_contorno_mayor(m)], -1, 255, -1)
    return out


def _ajustar_circulo(gris, area0, centro_mask, bbox, margen=12):
    """Transformada de Hough circular (Unidad 3.4) dentro del ROI de una moneda.
    Devuelve (cx, cy, r) en coordenadas de la imagen, o None si no hay círculo coherente."""
    H, W = gris.shape
    x, y, w, h = bbox
    # ROI = bounding box de la máscara más un margen, recortado a los bordes de la imagen
    x0, y0 = max(x - margen, 0), max(y - margen, 0)
    x1, y1 = min(x + w + margen, W), min(y + h + margen, H)
    roi = gris[y0:y1, x0:x1]

    # Radio equivalente de la máscara (el de un círculo con su misma área). Se busca
    # entre 0.85 y 1.3 veces ese radio: la máscara puede tener halo (radio real menor)
    # o mordiscos (radio real mayor).
    req = np.sqrt(area0 / np.pi)
    # dp=1: acumulador con la resolución de la imagen; minDist=5: poca restricción
    # (buscamos un solo círculo por ROI); param1=80: umbral alto del Canny interno;
    # param2=18: umbral de votos, BAJO a propósito: es permisivo porque se valida después.
    circles = cv2.HoughCircles(
        roi, cv2.HOUGH_GRADIENT, dp=1, minDist=5, param1=80, param2=18,
        minRadius=int(0.85 * req), maxRadius=int(1.3 * req))
    if circles is None:
        return None
    for cx, cy, r in circles[0]:          # OpenCV los devuelve ordenados por votos
        cx, cy = cx + x0, cy + y0         # pasar de coordenadas del ROI a las de la imagen
        # Validación: el centro del círculo debe estar cerca del centroide de la
        # máscara (a menos de 0.35 radios); descarta círculos espurios del grabado.
        if np.hypot(cx - centro_mask[0], cy - centro_mask[1]) < 0.35 * req:
            return cx, cy, r
    return None


def _cara_superior(L, m):
    """Segmenta SOLO la cara superior de un dado (sin sombra ni caras laterales).
    L: canal de luminancia de la imagen; m: máscara del dado (paso 3).
    Devuelve (máscara final, diccionario con etapas intermedias para graficar)."""
    # 1) 1er Otsu (Unidad 3.1): separa el dado de su sombra. Se aplica solo sobre los
    #    píxeles del objeto (vector Nx1), no sobre toda la imagen. Después se rellena el
    #    contorno externo para tapar los puntos negros -> 'sil' = silueta sin sombra.
    vals1 = L[m > 0].reshape(-1, 1)
    t1, _ = cv2.threshold(vals1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    claro1 = (L > t1) & (m > 0)
    sil = _rellenar_sin_esquina((claro1 * 255).astype(np.uint8))

    # 2) 2do Otsu sobre lo que quedó (el dado sin sombra): el histograma tiene dos
    #    grupos claros, las caras laterales (~150-190) y la cara superior (~215-225).
    #    La apertura 5x5 limpia tiras finas de las caras laterales.
    vals2 = L[claro1].reshape(-1, 1)
    t2, _ = cv2.threshold(vals2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cara_blanca = (((L > t2) & (sil > 0)) * 255).astype(np.uint8)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cara_blanca = cv2.morphologyEx(cara_blanca, cv2.MORPH_OPEN, k5)

    # 3) Puntos negros: son los huecos dentro de la silueta (silueta menos lo claro).
    #    Descriptor de forma (Unidad 7.2): circularidad = 4*pi*Área / Perímetro^2.
    #    Los puntos de la cara superior se ven redondos; los de las caras laterales
    #    salen aplastados por la perspectiva (circularidad baja). Umbral 0.72 y área
    #    mínima 25 px (descarta ruido). Se conservan solo los redondos.
    #    Limitación: un punto pegado al borde de la silueta no forma un hueco cerrado.
    huecos = cv2.subtract(sil, (claro1 * 255).astype(np.uint8))
    cnts_h, _ = cv2.findContours(huecos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    puntos_top = np.zeros_like(m)
    for c in cnts_h:
        area_p = cv2.contourArea(c)
        perim_p = cv2.arcLength(c, True)
        if perim_p > 0 and area_p > 25:
            circ_p = 4 * np.pi * area_p / (perim_p ** 2)
            if circ_p > 0.72:      # redondo = cara superior
                cv2.drawContours(puntos_top, [c], -1, 255, -1)

    # 4) Recuperar las esquinas: el 2do Otsu pierde el borde de la cara donde hay un
    #    punto cerca (queda oscuro). Se dilatan los puntos superiores (elemento 11x11) y
    #    se recortan con la silueta (así no se sale del dado). La unión con la cara
    #    blanca devuelve esas esquinas.
    k11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    esquinas_recuperadas = cv2.dilate(puntos_top, k11) & sil
    cara = cara_blanca | esquinas_recuperadas

    # 5) Envolvente convexa del componente mayor (Unidad 6.4): la cara es convexa y
    #    así se tapan los puntos negros que dejaron agujeros.
    final = np.zeros_like(m)
    cv2.drawContours(final, [cv2.convexHull(_contorno_mayor(cara))], -1, 255, -1)

    # 'nucleo' es el nombre histórico de esta clave: hoy guarda la cara completa
    # ANTES de la envolvente convexa (paso4.py la imprime con esa etiqueta)
    return final, dict(t1=t1, t2=t2, mask_t1=sil, nucleo=cara)


def refinar_objetos(seg, circ_dado=0.85):
    """PASO 4 (Punto A, etapa 2): separa monedas de dados por forma y refina cada máscara.
    - Moneda: círculo de Hough (respaldo: envolvente convexa).
    - Dado: cara superior (dos Otsu sobre L + puntos redondos + hull).
    Devuelve una lista de diccionarios, uno por objeto."""
    L = seg["lab"][:, :, 0]
    # Hough trabaja sobre la luminancia suavizada: el Gaussiano atenúa el grabado interno
    gris = cv2.GaussianBlur(L, (5, 5), 1.5)
    objetos = []
    for i in seg["keep"]:
        # Máscara binaria de este componente
        m = ((seg["labels"] == i) * 255).astype(np.uint8)
        cnt = _contorno_mayor(m)
        area0 = cv2.countNonZero(m)
        # Circularidad = 4*pi*Área/Perímetro^2 (Unidad 7.2): 1 para un círculo perfecto.
        # Las monedas dan 0.88-0.92 y los dados 0.81-0.82; el corte en 0.85 los separa
        circ0 = 4 * np.pi * area0 / cv2.arcLength(cnt, True) ** 2
        tipo = "dado" if circ0 < circ_dado else "moneda"
        bbox = seg["stats"][i, :4]       # x, y, ancho, alto
        centro = seg["cents"][i]

        if tipo == "moneda":
            mr = np.zeros_like(m)
            c = _ajustar_circulo(gris, area0, centro, bbox)
            if c is not None:
                # Se usa el borde REAL de la moneda y no la máscara con halo o mordiscos
                cx, cy, r = c
                cv2.circle(mr, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
                extra = dict(metodo="hough", r=r, centro_ajustado=(cx, cy))
            else:
                # Respaldo: envolvente convexa (tapa mordiscos, no corrige halo)
                cv2.drawContours(mr, [cv2.convexHull(cnt)], -1, 255, -1)
                extra = dict(metodo="hull", r=np.sqrt(cv2.countNonZero(mr) / np.pi),
                             centro_ajustado=tuple(centro))
        else:
            mr, extra = _cara_superior(L, m)

        objetos.append(dict(
            id=i, tipo=tipo, mask=mr, mask_orig=m,      # mask = refinada, mask_orig = del paso 3
            area0=area0, circ0=circ0, area=cv2.countNonZero(mr),
            centro=centro, bbox=bbox, **extra,           # extra: método/radio (monedas) o umbrales/etapas (dados)
        ))
    return objetos