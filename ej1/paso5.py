# =============================================================================
# PROBLEMA 1 - PASO 5: CLASIFICACIÓN DE MONEDAS (Punto B) Y VALOR DE LOS DADOS (Punto C)
# -----------------------------------------------------------------------------
# Parte de los objetos ya segmentados y refinados (pasos 3 y 4, segmentacion.py):
#   - cada moneda tiene su círculo de Hough (centro y radio reales)
#   - cada dado tiene la máscara de su cara superior
#
# Punto B - Monedas: se extraen dos descriptores por moneda (Unidad 7) y se
# clasifican con un árbol de decisión de dos umbrales (Unidad 7.5):
#   1) radio r (descriptor geométrico, 7.2): separa la moneda chica (10 centavos)
#   2) Δb* = b*(centro) - b*(anillo) (descriptor de color, 5.4 y 7.4): separa la
#      bimetálica de $1 (centro dorado y anillo plateado, o al revés) de la de
#      50 centavos (un solo metal, Δb* cercano a 0)
#
# Punto C - Dados: dentro de la cara superior se buscan los píxeles oscuros
# (puntos negros) y se cuentan con componentes conectados (Unidad 3.3).
# =============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segmentacion import cargar_imagen, segmentar, refinar_objetos

# Umbrales del árbol de decisión, fijados a partir del espacio de descriptores
# (ver el gráfico r vs Δb* de la figura). Se usan en la clasificación y en el gráfico.
R_CORTE = 37.5        # px a escala 1/4: por debajo, 10 centavos
DIFF_B_CORTE = 8.0    # Δb*: por encima, bimetálica de $1


def extraer_descriptores_moneda(o, lab, hsv):
    """Extrae los descriptores de una moneda:
    - geométrico: radio del círculo de Hough (Unidad 7.2)
    - de color: media de b* y de S en el centro y en el anillo exterior (Unidades 5.4 y 7.4)
    Devuelve un diccionario con los descriptores."""
    cx, cy = o["centro_ajustado"]    # centro del círculo de Hough (paso 4)
    r = o["r"]
    H, W = lab.shape[:2]

    # Distancia de cada píxel al centro de la moneda. np.ogrid arma vectores fila y
    # columna que se combinan por broadcasting (no crea dos matrices completas).
    yy, xx = np.ogrid[:H, :W]
    dist_centro = np.hypot(xx - cx, yy - cy)

    # Dos regiones concéntricas, en proporción al radio para que sirvan en
    # monedas de cualquier tamaño:
    #  - núcleo: hasta 0.55 r
    #  - anillo: entre 0.65 r y 0.92 r
    # Se deja una franja sin usar entre 0.55 r y 0.65 r (la unión de los dos metales
    # en la bimetálica) y se excluye el borde exterior (> 0.92 r), que mezcla
    # moneda, sombra y fondo.
    mask_nucleo = dist_centro <= (0.55 * r)
    mask_anillo = (dist_centro >= (0.65 * r)) & (dist_centro <= (0.92 * r))

    # Canales de color (en float32 para promediar y restar sin desbordes):
    #  - S (saturación, HSV): alta en los metales dorados
    #  - b* (CIELAB, eje azul-amarillo): alto en los metales dorados
    S = hsv[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)

    # Descriptor estadístico: media de cada canal en cada región
    s_nucleo = S[mask_nucleo].mean()
    s_anillo = S[mask_anillo].mean()
    b_nucleo = b[mask_nucleo].mean()
    b_anillo = b[mask_anillo].mean()

    # Diferencia centro - anillo: grande en la bimetálica de $1 (dos metales
    # distintos), cercana a 0 en las monedas de un solo metal.
    # Se usa Δb* para clasificar; Δs se calcula solo como dato de control (se imprime).
    diff_s = s_nucleo - s_anillo
    diff_b = b_nucleo - b_anillo

    return dict(
        r=r,
        s_nucleo=s_nucleo,
        s_anillo=s_anillo,
        diff_s=diff_s,
        diff_b=diff_b,
    )


def clasificar_monedas(monedas, lab, hsv, r_corte=R_CORTE, diff_b_corte=DIFF_B_CORTE):
    """Clasifica cada moneda en '10 centavos', '50 centavos' o '$1 (bimetálica)'
    con un árbol de decisión de dos niveles (Unidad 7.5):
        r < r_corte             -> 10 centavos (la más chica)
        si no, Δb* > diff_b_corte -> $1 (bimetálica)
        si no                   -> 50 centavos
    Agrega los descriptores y la clase al diccionario de cada moneda."""
    for o in monedas:
        desc = extraer_descriptores_moneda(o, lab, hsv)
        o.update(desc)
        if o["r"] < r_corte:
            o["clase"] = "10 centavos"
        elif o["diff_b"] > diff_b_corte:
            o["clase"] = "$1 (bimetálica)"
        else:
            o["clase"] = "50 centavos"
    return monedas


def contar_puntos_dado(d, L):
    """Punto C: cuenta los puntos negros de la cara superior de un dado.
    d: diccionario del dado (paso 4); L: canal de luminancia de la imagen.
    Agrega al diccionario el valor, la posición de cada punto y su máscara."""
    # Se erosiona la cara superior con un elemento 3x3 (Unidad 6.2): se le saca
    # 1 px de borde para no tomar como punto la sombra de las aristas.
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cara_int = cv2.erode(d["mask"], k3)

    # Puntos = píxeles oscuros dentro de la cara. Se reutiliza el 1er umbral de
    # Otsu del dado (t1, paso 4), que ya separaba lo oscuro de lo claro en ese dado.
    oscuros = (((L < d["t1"]) & (cara_int > 0)) * 255).astype(np.uint8)

    # Componentes conectados (Unidad 3.3): cada mancha oscura es un punto candidato.
    # Se descartan las de menos de 20 px de área (ruido).
    n_pips, labels_p, stats_p, cents_p = cv2.connectedComponentsWithStats(oscuros, connectivity=8)
    pips_validos = []
    for j in range(1, n_pips):  # j = 0 es el fondo
        area_p = stats_p[j, cv2.CC_STAT_AREA]
        if area_p >= 20:
            pips_validos.append((cents_p[j][0], cents_p[j][1], area_p))  # (x, y, área)

    # El valor del dado es la cantidad de puntos encontrados
    d["valor"] = len(pips_validos)
    d["pips"] = pips_validos
    d["mask_pips"] = oscuros
    return d


if __name__ == "__main__":
    # --- 1. Pipeline previo: segmentación (paso 3) y refinamiento (paso 4) ---
    img_bgr, small = cargar_imagen()                     # imagen a 1/4 (968x670)
    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)   # solo para mostrar con matplotlib
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)         # para el canal S
    seg = segmentar(small)
    lab = seg["lab"]
    L = lab[:, :, 0]                                     # luminancia, para los dados
    objetos = refinar_objetos(seg)

    monedas = [o for o in objetos if o["tipo"] == "moneda"]
    dados = [o for o in objetos if o["tipo"] == "dado"]

    # --- 2. Punto B: clasificación y conteo de monedas ---
    clasificar_monedas(monedas, lab, hsv)

    print("=== PUNTO B: CLASIFICACIÓN DE MONEDAS ===")
    print("id |   r  | b_nuc - b_ani | S_nuc - S_ani | Clase")
    for o in sorted(monedas, key=lambda x: x["r"]):
        print(f"{o['id']:2d} | {o['r']:4.1f} | {o['diff_b']:13.1f} | {o['diff_s']:13.1f} | {o['clase']}")

    # Conteo automático por clase
    conteo_monedas = {}
    for o in monedas:
        conteo_monedas[o["clase"]] = conteo_monedas.get(o["clase"], 0) + 1
    print("\nResumen de Monedas:")
    for clase, cant in conteo_monedas.items():
        print(f"  - {clase}: {cant}")
    print(f"  TOTAL MONEDAS: {len(monedas)}")

    # --- 3. Punto C: valor de la cara superior de cada dado y conteo ---
    print("\n=== PUNTO C: CONTEO DE DADOS ===")
    suma_dados = 0
    for d in dados:
        contar_puntos_dado(d, L)
        suma_dados += d["valor"]
        print(f"  - Dado id {d['id']}: valor cara superior = {d['valor']}")
    print(f"  CANTIDAD DE DADOS: {len(dados)} | SUMA DE PUNTOS: {suma_dados}")

    # --- 4. Visualización de los puntos B y C ---
    # Color de cada clase (nombre para matplotlib, hex de referencia)
    colores_clase = {
        "10 centavos": ("cyan", "#00FFFF"),
        "$1 (bimetálica)": ("lime", "#00FF00"),
        "50 centavos": ("gold", "#FFD700"),
    }

    fig, ax = plt.subplots(2, 2, figsize=(16, 12))

    # (0,0) Resultado final: cada moneda con el contorno y la etiqueta de su clase,
    # y cada dado con su valor
    ax[0, 0].imshow(small_rgb)
    for o in monedas:
        col_mpl, _ = colores_clase[o["clase"]]
        ax[0, 0].contour(o["mask"] > 0, levels=[0.5], colors=col_mpl, linewidths=1.8)
        etiqueta = "10c" if "10" in o["clase"] else ("$1" if "$1" in o["clase"] else "50c")
        ax[0, 0].text(o["centro"][0], o["centro"][1], etiqueta, color="black",
                      fontsize=9, weight="bold", ha="center", va="center",
                      bbox=dict(boxstyle="round,pad=0.15", fc=col_mpl, alpha=0.85))
    for d in dados:
        ax[0, 0].contour(d["mask"] > 0, levels=[0.5], colors="red", linewidths=2.0)
        ax[0, 0].text(d["centro"][0], d["centro"][1], f"Dado={d['valor']}", color="white",
                      fontsize=10, weight="bold", ha="center", va="center",
                      bbox=dict(boxstyle="round,pad=0.2", fc="red", alpha=0.85))
    ax[0, 0].set_title(f"Clasificación final: {conteo_monedas.get('10 centavos',0)}x10c, "
                       f"{conteo_monedas.get('$1 (bimetálica)',0)}x$1, "
                       f"{conteo_monedas.get('50 centavos',0)}x50c | Dados: {[d['valor'] for d in dados]}")
    ax[0, 0].axis("off")

    # (0,1) Espacio de descriptores r vs Δb*, con las dos fronteras de decisión.
    # Muestra que las tres clases quedan separadas por los umbrales elegidos.
    for clase, (col_mpl, _) in colores_clase.items():
        pts = [o for o in monedas if o["clase"] == clase]
        ax[0, 1].scatter([o["r"] for o in pts], [o["diff_b"] for o in pts],
                         label=f"{clase} (n={len(pts)})", s=90, edgecolors="k", c=col_mpl)
        for o in pts:
            ax[0, 1].annotate(str(o["id"]), (o["r"], o["diff_b"]), xytext=(4, 4),
                              textcoords="offset points", fontsize=8)
    ax[0, 1].axvline(R_CORTE, color="red", linestyle="--", label=f"Umbral radio ({R_CORTE} px)")
    ax[0, 1].axhline(DIFF_B_CORTE, color="blue", linestyle="--",
                     label=f"Umbral Δb* bimetálica ({DIFF_B_CORTE})")
    ax[0, 1].set_xlabel("Radio estimado r (píxeles)")
    ax[0, 1].set_ylabel("Δb* = b*(centro) - b*(anillo)")
    ax[0, 1].set_title("Espacio de descriptores y fronteras de decisión (Punto B)")
    ax[0, 1].legend()
    ax[0, 1].grid(True, alpha=0.3)

    # (1,0) y (1,1) Zoom de cada dado: contorno de la cara superior (verde) y
    # los puntos detectados, numerados
    for k, d in enumerate(dados[:2]):
        x, y, w, h = d["bbox"]
        mrg = 15
        ys, xs = slice(max(y - mrg, 0), y + h + mrg), slice(max(x - mrg, 0), x + w + mrg)
        ax[1, k].imshow(small_rgb[ys, xs])
        ax[1, k].contour(d["mask"][ys, xs] > 0, levels=[0.5], colors="lime", linewidths=1.5)
        # Las coordenadas de los puntos están en la imagen completa: se restan
        # el inicio del recorte para dibujarlas sobre el zoom
        for idx_p, (px, py, _) in enumerate(d["pips"], 1):
            ax[1, k].plot(px - xs.start, py - ys.start, "ro", markersize=8)
            ax[1, k].text(px - xs.start + 4, py - ys.start - 4, str(idx_p),
                          color="yellow", fontsize=11, weight="bold")
        ax[1, k].set_title(f"Dado id {d['id']} -> Valor detectado: {d['valor']} puntos")
        ax[1, k].axis("off")

    plt.tight_layout()
    plt.savefig("ej1/paso5_resultado.png", dpi=100)
    plt.show()   # sin tkinter solo da un warning; el resultado queda en el PNG