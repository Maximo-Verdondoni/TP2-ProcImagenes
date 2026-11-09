"""
TP2 - Problema 2 - Punto A: Detección automática de la placa patente.

Pipeline general de detectar_patente():
    1. Escala de grises.
    2. Top-Hat morfológico: resalta estructuras claras y finas (los trazos de
       las letras) sobre un fondo más oscuro, y apaga zonas claras grandes.
    3. Sobel en X: la patente tiene muchos bordes VERTICALES juntos
       (los costados de cada carácter), cosa que no pasa en la mayor parte del auto.
    4. Umbralizado del gradiente + clausura horizontal: "pega" los bordes de los
       caracteres en un único bloque rectangular, que es el candidato a patente.
    5. Componentes conectados + filtro geométrico (relación de aspecto, tamaño,
       extent, posición): se queda con los bloques con forma de patente.
    6. Validación interna: dentro de cada candidato se cuentan los componentes
       con forma de carácter alineados en una fila. Gana el candidato con
       ~6 caracteres alineados y mayor energía de bordes.

Para hacerlo más robusto se prueban 2 umbrales x 2 tamaños de clausura y se
juntan todos los candidatos antes de elegir el ganador.
"""

import glob
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Carpeta con las 12 imágenes de entrada (img01.png ... img12.png)
CARPETA = "ej2/Patentes"
archivos = sorted(glob.glob(os.path.join(CARPETA, "img*.png")))
assert len(archivos) > 0, f"No se encontraron archivos img*.png en '{CARPETA}'"


def _contar_caracteres_internos(roi_gray):
    """
    Valida un candidato a patente contando cuántos componentes con forma de
    carácter (letra/número) hay alineados en una misma fila horizontal.

    Parámetros
    ----------
    roi_gray : recorte en escala de grises del candidato.

    Devuelve
    --------
    (cantidad, uniformidad)
        cantidad    : tamaño del grupo más grande de caracteres alineados.
        uniformidad : 1 / (1 + desvío de las alturas del grupo). Vale 1 si todos
                      miden lo mismo y baja cuanto más dispares sean.
    """
    rh, rw = roi_gray.shape

    # Recortes demasiado chicos no pueden contener una patente legible
    if rh < 8 or rw < 25:
        return 0, 0.0

    # Filtro de contraste: una patente tiene letras blancas sobre chapa negra,
    # por lo que la diferencia entre píxeles claros (percentil 90) y oscuros
    # (percentil 15) es grande. Veredas y pisos grises son parejos y se descartan.
    if np.percentile(roi_gray, 90) - np.percentile(roi_gray, 15) < 45:
        return 0, 0.0

    # Estiramiento de contraste del recorte a [0, 255] (Unidad 2.2) y
    # umbralizado de Otsu (Unidad 3.1): las letras quedan en blanco
    roi_norm = cv2.normalize(roi_gray, None, 0, 255, cv2.NORM_MINMAX)
    _, bin_roi = cv2.threshold(roi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Top-Hat con elemento estructural de línea horizontal (Unidad 6.4):
    # Top-Hat = binaria - apertura. La apertura con una línea larga conserva solo
    # lo que tiene tramos horizontales largos (marco, carrocería, paragolpes), así
    # que al restarla se eliminan esas estructuras y las letras quedan sueltas.
    largo = max(15, int(0.35 * rw))
    se_linea = cv2.getStructuringElement(cv2.MORPH_RECT, (largo, 1))
    bin_roi = cv2.morphologyEx(bin_roi, cv2.MORPH_TOPHAT, se_linea)

    # Componentes conectados (Unidad 3.3): cada mancha blanca es un posible carácter
    n, _, stats, cents = cv2.connectedComponentsWithStats(bin_roi, connectivity=8)

    # Filtro geométrico de carácter individual (Unidad 7.2):
    #   - 0.12 <= ancho/alto <= 0.95 : más alto que ancho
    #   - h >= 7 px                  : altura mínima absoluta (descarta ruido)
    #   - h/rh <= 0.92               : no ocupa todo el alto del recorte (marco lateral)
    #   - area >= 8                  : descarta puntitos
    chars = []
    for j in range(1, n):  # j = 0 es el fondo
        x, y, w, h, area = stats[j]
        ar_c = w / float(h)
        rel_h = h / float(rh)
        if 0.12 <= ar_c <= 0.95 and h >= 7 and rel_h <= 0.92 and area >= 8:
            chars.append((cents[j][0], cents[j][1], h, w))  # (cx, cy, alto, ancho)

    # Con menos de 3 componentes no tiene sentido buscar una fila
    if len(chars) < 3:
        return len(chars), 0.0

    # Búsqueda de la fila: tomando cada carácter como referencia, se cuentan los
    # que están a la misma altura (centroide y similar) y tienen alto parecido
    # (tolerancia del 35% del alto). Nos quedamos con el grupo más grande.
    mejor_grupo = 0
    uniformidad = 0.0
    for _, cy_ref, h_ref, _ in chars:
        alineados = [
            c for c in chars
            if abs(c[1] - cy_ref) < 0.35 * h_ref and abs(c[2] - h_ref) < 0.35 * h_ref
        ]
        if len(alineados) > mejor_grupo:
            mejor_grupo = len(alineados)
            alturas = [c[2] for c in alineados]
            uniformidad = 1.0 / (1.0 + np.std(alturas))

    return mejor_grupo, uniformidad


def detectar_patente(img_bgr):
    """
    Punto 2.A: detecta la patente en una imagen de un vehículo.

    Parámetros
    ----------
    img_bgr : imagen color (BGR, como la lee cv2.imread).

    Devuelve
    --------
    dict con las etapas intermedias (para poder mostrarlas) y el resultado:
        gray       : imagen en escala de grises
        tophat     : resultado del Top-Hat
        abs_gx     : módulo del gradiente horizontal suavizado
        mask_bin   : umbral de |Gx| de la combinación que dio el ganador
        mask_morf  : máscara tras clausura + apertura de esa combinación
        bbox       : (x, y, w, h) de la patente detectada, o None
        candidatos : lista de (score, n_chars, bbox, mask_morf, mask_bin)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    # 1. Top-Hat morfológico en escala de grises (Unidad 6.4).
    #    Top-Hat = imagen - apertura. Con un elemento de 15x5 (más grande que el
    #    grosor de un trazo) la apertura borra los trazos finos y claros; al
    #    restarla quedan resaltadas las letras y se apagan zonas claras extensas
    #    (capó, carrocería, cielo).
    se_th = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se_th)

    # 2. Gradiente horizontal con Sobel en X (Unidad 3.2): responde a bordes
    #    VERTICALES, que en la patente son muy densos (dos por cada carácter).
    #    Se toma el módulo y se suaviza con un Gaussiano (Unidad 2.4) para que
    #    los bordes de caracteres vecinos se acerquen entre sí.
    gx = cv2.Sobel(tophat, cv2.CV_32F, 1, 0, ksize=3)
    abs_gx = cv2.convertScaleAbs(np.abs(gx))
    abs_gx = cv2.GaussianBlur(abs_gx, (5, 3), 0)

    # 3. Umbralizado + morfología, probando 2 x 2 combinaciones (Unidades 3.1 y 6.3).
    #    - Umbral: Otsu y 1.35 * Otsu. El más alto deja menos bordes y evita que,
    #      en autos con parrilla pegada a la patente (img04, img06), ambas se
    #      fusionen en un solo bloque.
    #    - Clausura horizontal de 15 o 21 px de ancho: une los bordes de los
    #      caracteres en un rectángulo continuo.
    #    - Apertura de 5x3: elimina restos chicos y conexiones finas.
    t_otsu, _ = cv2.threshold(abs_gx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidatos = []

    for factor_t in (1.0, 1.35):
        _, mask_bin = cv2.threshold(abs_gx, int(min(250, t_otsu * factor_t)), 255,
                                    cv2.THRESH_BINARY)
        for ancho_close in (15, 21):
            se_close = cv2.getStructuringElement(cv2.MORPH_RECT, (ancho_close, 3))
            se_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            m_morf = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, se_close)
            m_morf = cv2.morphologyEx(m_morf, cv2.MORPH_OPEN, se_open)

            # 4. Componentes conectados (Unidad 3.3): cada bloque es un candidato
            n, _, stats, _ = cv2.connectedComponentsWithStats(m_morf, connectivity=8)
            for i in range(1, n):  # i = 0 es el fondo
                x, y, w, h, area = stats[i]
                ar = w / float(h)              # relación de aspecto
                extent = area / float(w * h)   # qué tan "lleno" está el rectángulo

                # 5. Filtro geométrico de placa (Unidad 7.2):
                #    - relación de aspecto entre 1.65 y 5.5 (rectángulo apaisado)
                #    - ancho entre 35 y 122 px y alto entre 11 y 65 px
                #    - extent > 0.35: el bloque debe llenar bien su rectángulo
                #    - y > 0.25*H: descarta estructuras altas (techo, parabrisas)
                if (
                    1.65 <= ar <= 5.5
                    and 35 <= w <= 122
                    and 11 <= h <= 65
                    and extent > 0.35
                    and y > 0.25 * H
                ):
                    # Se agrega un pequeño margen al recuadro para que los
                    # caracteres de los extremos queden completos al validar
                    px, py = max(2, int(0.05 * w)), max(2, int(0.15 * h))
                    x0, y0 = max(0, x - px), max(0, y - py)
                    x1, y1 = min(W, x + w + px), min(H, y + h + py)

                    roi_g = gray[y0:y1, x0:x1]
                    roi_gx = abs_gx[y0:y1, x0:x1]

                    # 6. Validación interna: ¿hay una fila de caracteres adentro?
                    n_chars, unif = _contar_caracteres_internos(roi_g)

                    # Energía media de bordes verticales dentro del candidato
                    energia_gx = float(np.mean(roi_gx))

                    # Puntaje: la cantidad de caracteres alineados (tope 6) pesa
                    # con exponente 2.5, así un candidato con 6 caracteres le gana
                    # con amplitud a un logo o una parrilla (que dan 1 o 2), aunque
                    # estos tengan más energía de bordes. Si n_chars = 0, score = 0.
                    bonus_chars = (min(n_chars, 6) ** 2.5) * (1.0 + unif)
                    score = bonus_chars * energia_gx
                    candidatos.append((score, n_chars, (x0, y0, x1 - x0, y1 - y0),
                                       m_morf, mask_bin))

    # Elección del ganador: el candidato de mayor puntaje entre todas las combinaciones
    if candidatos:
        mejor = max(candidatos, key=lambda c: c[0])
        bbox_final = mejor[2]
        mask_morf_final = mejor[3]
        mask_bin_final = mejor[4]
    else:
        bbox_final = None
        mask_morf_final = np.zeros_like(gray)
        mask_bin_final = np.zeros_like(gray)

    return dict(
        gray=gray,
        tophat=tophat,
        abs_gx=abs_gx,
        mask_bin=mask_bin_final,
        mask_morf=mask_morf_final,
        bbox=bbox_final,
        candidatos=candidatos,
    )


if __name__ == "__main__":
    # Carpeta de salida relativa al .py, así funciona desde donde lo corras
    CARPETA_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_pasos")
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    resultados = []

    # --- Figura por imagen con todas las etapas del punto A ---
    for ruta in archivos:
        nombre = os.path.basename(ruta)
        img_bgr = cv2.imread(ruta)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # matplotlib espera RGB
        res = detectar_patente(img_bgr)
        resultados.append((nombre, img_rgb, res))

        # Informe por consola del resultado de cada imagen
        if res["bbox"] is not None:
            mejor = max(res["candidatos"], key=lambda c: c[0])
            x, y, w, h = res["bbox"]
            print(f"{nombre:10s}: {len(res['candidatos']):2d} candidatos | ganador: "
                  f"score={mejor[0]:7.1f}, caracteres={mejor[1]} | "
                  f"bbox=({x},{y},{w},{h}), AR={w/h:.2f}")
            recorte = img_rgb[y:y + h, x:x + w]
        else:
            print(f"{nombre:10s}: {len(res['candidatos']):2d} candidatos | SIN DETECTAR")
            recorte = np.zeros((10, 30, 3), np.uint8)

        # Figura de 2x4 con cada etapa del procesamiento.
        # Los paneles 5 y 6 corresponden a la combinación (umbral, clausura) de la
        # que salió el ganador; el panel 7 muestra los candidatos de todas las
        # combinaciones (amarillo) y el elegido (verde).
        fig, axes = plt.subplots(2, 4, figsize=(20, 8))
        ax = axes.ravel()
        ax[0].imshow(img_rgb);                       ax[0].set_title("1. Original")
        ax[1].imshow(res["gray"], cmap="gray");      ax[1].set_title("2. Escala de grises")
        ax[2].imshow(res["tophat"], cmap="gray");    ax[2].set_title("3. Top-Hat (15x5)")
        ax[3].imshow(res["abs_gx"], cmap="gray");    ax[3].set_title("4. |Gx| (Sobel) suavizado")
        ax[4].imshow(res["mask_bin"], cmap="gray");  ax[4].set_title("5. Umbral sobre |Gx|")
        ax[5].imshow(res["mask_morf"], cmap="gray"); ax[5].set_title("6. Clausura + apertura")
        ax[6].imshow(img_rgb)
        for _, _, (cx, cy, cw, ch), _, _ in res["candidatos"]:
            ax[6].add_patch(plt.Rectangle((cx, cy), cw, ch, fill=False,
                                          edgecolor="yellow", linewidth=1))
        if res["bbox"] is not None:
            ax[6].add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                          edgecolor="lime", linewidth=2.5))
        ax[6].set_title(f"7. Candidatos ({len(res['candidatos'])}) y ganador")
        ax[7].imshow(recorte);                       ax[7].set_title("8. Patente segmentada")
        for a in ax:
            a.axis("off")
        fig.suptitle(nombre)
        plt.tight_layout()
        plt.savefig(os.path.join(CARPETA_SALIDA, f"paso1_etapas_{nombre}"), dpi=110)
        plt.close(fig)

    # --- Grilla resumen de las 12 detecciones ---
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    for ax, (nombre, img_rgb, res) in zip(axes.ravel(), resultados):
        ax.imshow(img_rgb)
        if res["bbox"] is not None:
            x, y, w, h = res["bbox"]
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=2.5))
            ax.set_title(f"{nombre}: {w}x{h} (AR={w/h:.1f})", fontsize=11)
        else:
            ax.set_title(f"{nombre}: SIN DETECTAR", color="red", fontsize=11)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_SALIDA, "paso1_deteccion_12_patentes.png"), dpi=120)
    plt.close(fig)
    print(f"\nFiguras guardadas en {CARPETA_SALIDA}")