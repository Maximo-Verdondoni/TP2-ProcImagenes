"""
TP2 - Problema 2 - Punto B: Segmentación de los caracteres de la patente.

Parte del recorte de la patente que devuelve detectar_patente() (paso1.py)
y lo procesa así:
    1.  Escalado x3: a la resolución original las letras miden ~10 px de alto y
        la separación entre letra y marco es de 1 px; agrandada se conserva mejor.
    2.  Enderezado: se mide la inclinación con Hough sobre los bordes del marco
        y se rota la patente para dejarla horizontal.
    3.  Realce con Top-Hat en escala de grises: atenúa el marco blanco grueso.
    4.  Umbralizado de Otsu: letras blancas sobre chapa negra.
    5.  Limpieza morfológica: Top-Hat binario con línea horizontal (borra el
        marco) y aperturas chicas (cortan puentes finos letra-marco).
    6.  Se eliminan los objetos que tocan los bordes laterales del recorte.
    7.  Se recorta la banda vertical donde están las letras (perfil de Sobel X).
    8.  Se cortan pares de caracteres unidos por el valle de su perfil vertical.
    9.  Componentes conectados + filtro geométrico de carácter.
    10. Se elige la fila principal de caracteres alineados (máximo 6).
    11. Ajuste fino de cada carácter a la franja común de la fila.
    12. Respaldo para patentes sin marco visible: si Hough no encontró líneas,
        se estima el ángulo con los centroides de los caracteres y se repite todo.
"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Reutilizamos la detección del punto A (cambiar 'paso1' si tu archivo se llama distinto)
from paso1 import detectar_patente, archivos

ESC = 3  # factor de escalado del recorte de la patente (los umbrales en px se multiplican por él)


def estimar_inclinacion(roi):
    """
    Estima la inclinación de la patente a partir de las líneas largas casi
    horizontales de su marco (bordes superior e inferior).

    Usa Canny para obtener bordes (Unidad 3.2) y la Transformada de Hough
    probabilística para encontrar segmentos de recta (Unidad 3.4).

    Devuelve el ángulo en grados (mediana de los segmentos encontrados), o
    None si no se encontró ninguna línea (por ejemplo, patentes sin marco claro).
    """
    rw = roi.shape[1]
    bordes = cv2.Canny(roi, 50, 150)

    # Solo interesan líneas largas: al menos el 40% del ancho del recorte.
    # El umbral de votos también es proporcional al ancho.
    lineas = cv2.HoughLinesP(bordes, 1, np.pi / 180, threshold=int(0.25 * rw),
                             minLineLength=int(0.4 * rw), maxLineGap=5)
    if lineas is None:
        return None

    # reshape(-1, 4) porque según la versión de OpenCV el resultado viene
    # con forma (N, 1, 4) o (N, 4)
    angulos = []
    for x1, y1, x2, y2 in lineas.reshape(-1, 4):
        ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Solo líneas casi horizontales (bordes superior/inferior del marco);
        # se descartan los costados y otras líneas del recorte
        if abs(ang) < 25:
            angulos.append(ang)

    # La mediana es robusta a alguna línea suelta con un ángulo distinto
    return float(np.median(angulos)) if angulos else None


def segmentar_caracteres(gray, bbox, angulo_forzado=None):
    """
    Punto 2.B: segmenta los caracteres de la patente detectada.

    Parámetros
    ----------
    gray           : imagen completa en escala de grises.
    bbox           : (x, y, w, h) de la patente, devuelto por detectar_patente().
    angulo_forzado : si se indica, se usa este ángulo en lugar de Hough. Lo usa
                     la propia función al re-segmentar con el ángulo de los
                     centroides (paso 12).

    Devuelve
    --------
    dict con las etapas intermedias (para mostrarlas) y el resultado:
        roi              : recorte original de la patente
        roi_rot          : recorte escalado y enderezado
        angulo, metodo   : ángulo aplicado y cómo se obtuvo
                           ("Hough", "centroides" o "sin líneas")
        roi_norm         : imagen realzada y normalizada que entra a Otsu
        bin_otsu         : binaria de Otsu
        bin_sin_marco    : binaria final limpia (solo caracteres)
        labels           : etiquetas de componentes conectados
        chars_candidatos : componentes que pasaron el filtro de carácter
        chars            : caracteres finales (x, y, w, h, cy), ordenados de izq. a der.
    """
    x, y, w, h = bbox
    roi_orig = gray[y:y + h, x:x + w]

    # 1. Escalado x3 con interpolación cúbica (Unidad 2.1).
    #    A resolución original la separación oscura entre letra y marco mide ~1 px
    #    y el umbralizado la "come"; agrandada se conserva.
    roi = cv2.resize(roi_orig, None, fx=ESC, fy=ESC, interpolation=cv2.INTER_CUBIC)
    rh, rw = roi.shape

    # 2. Enderezado (Unidades 3.4 y 4.4).
    #    El ángulo sale de Hough sobre el marco; si Hough no encuentra líneas se
    #    usa 0° y más abajo (paso 12) se corrige con los centroides de los caracteres.
    if angulo_forzado is not None:
        angulo, metodo = angulo_forzado, "centroides"
    else:
        angulo, metodo = estimar_inclinacion(roi), "Hough"
        if angulo is None:
            angulo, metodo = 0.0, "sin líneas"

    # Rotación alrededor del centro del recorte. BORDER_REPLICATE rellena las
    # esquinas que quedan vacías al rotar con los píxeles del borde (en lugar de
    # negro), para no crear bordes artificiales.
    M = cv2.getRotationMatrix2D((rw / 2, rh / 2), angulo, 1.0)
    roi_rot = cv2.warpAffine(roi, M, (rw, rh), flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)

    # 3. Realce ANTES de Otsu: Top-Hat en escala de grises (Unidad 6.4).
    #    El trazo de una letra mide ~3*ESC px de ancho. Con una línea horizontal
    #    de 12*ESC px, la apertura conserva las zonas claras anchas (marco grueso)
    #    y el Top-Hat (imagen - apertura) se queda con los trazos finos: las letras.
    se_gris = cv2.getStructuringElement(cv2.MORPH_RECT, (12 * ESC, 1))
    roi_th = cv2.morphologyEx(roi_rot, cv2.MORPH_TOPHAT, se_gris)

    # Combinación ponderada: 0.45 * imagen enderezada + 0.85 * Top-Hat.
    # Se le da más peso al Top-Hat para que las letras resalten sobre el marco,
    # pero se conserva parte de la imagen original para no perder la forma de
    # las letras. Luego se normaliza a [0, 255] (Unidad 2.2).
    roi_realzado = cv2.addWeighted(roi_rot, 0.45, roi_th, 0.85, 0)
    roi_norm = cv2.normalize(roi_realzado, None, 0, 255, cv2.NORM_MINMAX)

    # 4. Umbralizado de Otsu sobre la imagen realzada (Unidad 3.1):
    #    letras en blanco, chapa en negro
    t_otsu, bin_otsu = cv2.threshold(roi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Eliminación del marco (Unidades 6.3 y 6.4)
    #    5a. Top-Hat binario con línea horizontal = binaria - apertura.
    #        La apertura conserva lo que tiene tramos horizontales de más del 18%
    #        del ancho (barras del marco, carrocería); al restarla, desaparecen.
    #        Funciona bien porque la patente ya está enderezada.
    largo = max(12, int(0.18 * rw))
    se_linea = cv2.getStructuringElement(cv2.MORPH_RECT, (largo, 1))
    bin_sin_marco = cv2.morphologyEx(bin_otsu, cv2.MORPH_TOPHAT, se_linea)

    #    5b. Dos aperturas chicas para cortar "puentes" de 1-2 px entre letras y
    #        marco o entre letras (D en img03, 9 en img11, 22 en img10):
    #        - vertical 1x5: elimina conexiones horizontales finas
    #        - horizontal 3x1: elimina conexiones verticales finas
    #        Los trazos de las letras son más gruesos y sobreviven.
    bin_sin_marco = cv2.morphologyEx(bin_sin_marco, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    bin_sin_marco = cv2.morphologyEx(bin_sin_marco, cv2.MORPH_OPEN,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))

    # 6. Se borran los objetos que tocan el borde izquierdo o derecho del recorte:
    #    son restos del marco lateral, de la carrocería o manchas (como en img03),
    #    nunca caracteres, que están siempre dentro de la chapa.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_sin_marco, connectivity=8)
    for j in range(1, n):  # j = 0 es el fondo
        cx, cy, cw, ch, _ = stats[j]
        if cx <= 2 or (cx + cw) >= (rw - 2):
            bin_sin_marco[labels == j] = 0

    # 7. Banda vertical donde están las letras, a partir del perfil de filas de
    #    |Sobel X| (Unidad 3.2): las filas que cruzan los caracteres tienen muchos
    #    bordes verticales; las del marco superior/inferior, casi ninguno.
    #    Se usa solo el 70% central del ancho para no contar los costados del marco.
    gx_roi = np.abs(cv2.Sobel(roi_rot, cv2.CV_32F, 1, 0, ksize=3))
    perfil_filas = gx_roi[:, int(0.15 * rw):int(0.85 * rw)].mean(axis=1)
    # Filas con más del 35% del pico de energía = franja de los caracteres
    filas_activas = np.where(perfil_filas > 0.35 * perfil_filas.max())[0]
    if len(filas_activas) > 5:
        # Se agrega un margen de 2*ESC px arriba y abajo y se borra todo lo de afuera
        y_top_banda = max(0, int(filas_activas[0]) - 2 * ESC)
        y_bot_banda = min(rh, int(filas_activas[-1]) + 2 * ESC)
        bin_sin_marco[:y_top_banda, :] = 0
        bin_sin_marco[y_bot_banda:, :] = 0

    # 8. Separación de caracteres unidos (ej. '22' en img10).
    #    Un carácter mide ~11% del ancho de la patente; si un componente mide más
    #    de 1.55 veces eso, probablemente son dos letras pegadas. Se calcula su
    #    perfil vertical (suma de píxeles blancos por columna) y se corta en la
    #    columna con menos píxeles (el "valle"), buscando solo en la mitad central.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bin_sin_marco, connectivity=8)
    w_esperado = 0.11 * rw
    for j in range(1, n):
        cx, cy, cw, ch, _ = stats[j]
        if ch >= 6 * ESC and cw > 1.55 * w_esperado:
            sub = bin_sin_marco[cy:cy + ch, cx:cx + cw]
            perfil_v = sub.sum(axis=0)
            ini, fin = int(0.25 * cw), int(0.75 * cw)
            if fin > ini:
                col_corte = ini + int(np.argmin(perfil_v[ini:fin]))
                # Se borra una franja de 3 columnas en el punto de corte
                bin_sin_marco[cy:cy + ch, max(0, cx + col_corte - 1):min(rw, cx + col_corte + 2)] = 0

    # 9. Re-etiquetado (Unidad 3.3) y filtro geométrico de carácter (Unidad 7.2):
    #    - 0.12 <= ancho/alto <= 0.95 : más alto que ancho
    #    - alto >= 6*ESC px           : altura mínima (descarta ruido)
    #    - alto/rh <= 0.90            : no ocupa todo el alto del recorte
    #    - ancho/rw <= 0.22           : un carácter no es más ancho que ~1/5 de la patente
    #    - área >= 7*ESC^2            : descarta puntitos
    #    Lo que no pasa el filtro se borra de la binaria y de las etiquetas,
    #    así las figuras muestran solo lo que quedó.
    n, labels, stats, cents = cv2.connectedComponentsWithStats(bin_sin_marco, connectivity=8)
    chars = []
    for j in range(1, n):
        cx, cy, cw, ch, area = stats[j]
        if (0.12 <= cw / ch <= 0.95
                and ch >= 6 * ESC and ch / rh <= 0.90
                and cw / rw <= 0.22
                and area >= 7 * ESC ** 2):
            # (x, y, ancho, alto, centroide y, etiqueta)
            chars.append((cx, cy, cw, ch, cents[j][1], j))
        else:
            bin_sin_marco[labels == j] = 0
            labels[labels == j] = 0

    # 10. Fila principal de caracteres alineados.
    #     Tomando cada carácter como referencia, se juntan los que tienen el
    #     centroide a la misma altura y un alto parecido (tolerancia del 32%).
    #     El puntaje de la fila premia la cantidad de caracteres y penaliza
    #     levemente la dispersión de alturas.
    mejor = []
    mejor_score = -1.0
    for c_ref in chars:
        h_ref, cy_ref = c_ref[3], c_ref[4]
        alineados = [
            c for c in chars
            if abs(c[4] - cy_ref) < 0.32 * h_ref and abs(c[3] - h_ref) < 0.32 * h_ref
        ]
        score_fila = len(alineados) - 0.08 * np.std([c[3] for c in alineados])
        if score_fila > mejor_score:
            mejor_score = score_fila
            mejor = alineados

    # Una patente tiene 6 caracteres: si sobró algo (un resto de marco en una
    # punta), nos quedamos con los 6 de altura más cercana a la mediana
    if len(mejor) > 6:
        h_med = np.median([c[3] for c in mejor])
        mejor = sorted(mejor, key=lambda c: abs(c[3] - h_med))[:6]

    # 11. Ajuste fino a la franja común de la fila (Unidad 7.2).
    #     Todos los caracteres ocupan la misma franja vertical: se toma la mediana
    #     de los bordes superior e inferior (robusta a un carácter "raro").
    #     Si una letra quedó unida a una mancha (D en img03), lo que sobresale
    #     por arriba o por abajo se descarta y se recalcula su bounding box.
    #     Además, así todos los recortes quedan del mismo alto.
    if len(mejor) >= 3:
        y_top = int(np.median([c[1] for c in mejor]))
        y_bot = int(np.median([c[1] + c[3] for c in mejor]))
        ajustados = []
        for cx, cy, cw, ch, cyc, j in mejor:
            comp = labels == j
            dentro = comp.copy()
            dentro[:y_top, :] = False
            dentro[y_bot:, :] = False
            # Se borra de la binaria y de las etiquetas lo que quedó fuera de la franja
            fuera = comp & ~dentro
            bin_sin_marco[fuera] = 0
            labels[fuera] = 0
            # Nuevo ancho: columnas que todavía tienen píxeles del componente
            xs = np.where(dentro.any(axis=0))[0]
            if len(xs) == 0:
                continue
            ajustados.append((int(xs[0]), y_top, int(xs[-1] - xs[0] + 1), y_bot - y_top, cyc, j))
        mejor = ajustados

    # 12. Respaldo: si Hough no encontró el marco (patente sin borde blanco, img12),
    #     se estima el ángulo con la recta que pasa por los centroides de los
    #     caracteres (ajuste por mínimos cuadrados con polyfit) y se vuelve a
    #     segmentar con ese ángulo. La llamada recursiva entra con
    #     metodo = "centroides", así que no se repite más de una vez.
    if metodo == "sin líneas" and len(mejor) >= 4:
        xs_c = [c[0] + c[2] / 2 for c in mejor]   # centro x de cada carácter
        ys_c = [c[4] for c in mejor]              # centroide y de cada carácter
        pendiente = np.polyfit(xs_c, ys_c, 1)[0]
        ang_c = float(np.degrees(np.arctan(pendiente)))
        if abs(ang_c) > 1.0:  # si la inclinación es mínima no vale la pena rotar
            return segmentar_caracteres(gray, bbox, angulo_forzado=ang_c)

    # Orden final de izquierda a derecha; se descarta la etiqueta (ya no se usa)
    mejor = sorted([c[:5] for c in mejor], key=lambda c: c[0])

    return dict(roi=roi_orig, roi_rot=roi_rot, angulo=angulo, metodo=metodo, roi_norm=roi_norm,
                bin_otsu=bin_otsu, bin_sin_marco=bin_sin_marco, labels=labels,
                chars_candidatos=chars, chars=mejor)


if __name__ == "__main__":
    # Carpeta de salida relativa al .py, así funciona desde donde lo corras
    CARPETA_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_pasos")
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    for ruta in archivos:
        nombre = os.path.basename(ruta)
        img_bgr = cv2.imread(ruta)

        # Punto A: detección de la patente (paso1.py)
        det = detectar_patente(img_bgr)
        if det["bbox"] is None:
            print(f"{nombre}: sin patente detectada, no se segmenta")
            continue

        # Punto B: segmentación de caracteres
        seg = segmentar_caracteres(det["gray"], det["bbox"])
        print(f"{nombre}: ángulo={seg['angulo']:+5.1f}° ({seg['metodo']}) | "
              f"{len(seg['chars_candidatos'])} componentes tipo carácter "
              f"-> {len(seg['chars'])} caracteres alineados")

        # Tira con los caracteres recortados, todos llevados a 40 px de alto y
        # separados por 6 px negros, para verlos juntos
        recortes = []
        for cx, cy, cw, ch, _ in seg["chars"]:
            r = seg["bin_sin_marco"][cy:cy + ch, cx:cx + cw]
            r = cv2.resize(r, (max(1, int(cw * 40 / ch)), 40), interpolation=cv2.INTER_NEAREST)
            recortes += [r, np.zeros((40, 6), np.uint8)]
        tira = np.hstack(recortes) if recortes else np.zeros((40, 40), np.uint8)

        # Figura con todas las etapas del procesamiento
        fig, axes = plt.subplots(1, 7, figsize=(26, 3.5))
        axes[0].imshow(seg["roi"], cmap="gray");             axes[0].set_title("1. Recorte patente")
        axes[1].imshow(seg["roi_rot"], cmap="gray");         axes[1].set_title(f"2. x{ESC} y enderezado ({seg['angulo']:+.1f}°, {seg['metodo']})")
        axes[2].imshow(seg["bin_otsu"], cmap="gray");        axes[2].set_title("3. Realce + Otsu")
        axes[3].imshow(seg["bin_sin_marco"], cmap="gray");   axes[3].set_title("4. Binaria limpia (sin marco)")
        axes[4].imshow(seg["labels"], cmap="nipy_spectral"); axes[4].set_title("5. Componentes")
        axes[5].imshow(seg["roi_rot"], cmap="gray")
        for cx, cy, cw, ch, _ in seg["chars"]:
            axes[5].add_patch(plt.Rectangle((cx, cy), cw, ch, fill=False,
                                            edgecolor="lime", linewidth=1.5))
        axes[5].set_title(f"6. Caracteres: {len(seg['chars'])}")
        axes[6].imshow(tira, cmap="gray");                   axes[6].set_title("7. Segmentados")
        for a in axes:
            a.axis("off")
        fig.suptitle(nombre)
        plt.tight_layout()
        plt.savefig(os.path.join(CARPETA_SALIDA, f"paso2_caracteres_{nombre}"), dpi=110)
        plt.close(fig)