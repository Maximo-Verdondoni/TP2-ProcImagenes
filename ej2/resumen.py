import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Reutilizamos el punto A (paso1) y el punto B (paso2)
from paso1 import detectar_patente, archivos
from paso2 import segmentar_caracteres


def tira_caracteres(seg, alto=40, sep=6):
    """Concatena horizontalmente los caracteres segmentados, todos llevados al mismo alto."""
    recortes = []
    for cx, cy, cw, ch, _ in seg["chars"]:
        r = seg["bin_sin_marco"][cy:cy + ch, cx:cx + cw]
        r = cv2.resize(r, (max(1, int(cw * alto / ch)), alto), interpolation=cv2.INTER_NEAREST)
        recortes += [r, np.zeros((alto, sep), np.uint8)]
    return np.hstack(recortes) if recortes else np.zeros((alto, alto), np.uint8)


if __name__ == "__main__":
    CARPETA_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_pasos")
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # Grilla 3x4: cada imagen ocupa 2 filas (arriba la patente, abajo sus caracteres)
    fig, axes = plt.subplots(6, 4, figsize=(20, 11),
                             gridspec_kw={"height_ratios": [3, 1] * 3})

    n_detectadas, n_seis = 0, 0
    print(f"{'Imagen':10s} | {'Patente':8s} | {'Caract.':7s} | Enderezado")
    print("-" * 50)

    for k, ruta in enumerate(archivos):
        nombre = os.path.basename(ruta)
        fila, col = 2 * (k // 4), k % 4
        ax_pat, ax_car = axes[fila, col], axes[fila + 1, col]
        ax_pat.axis("off"); ax_car.axis("off")

        img_bgr = cv2.imread(ruta)
        det = detectar_patente(img_bgr)

        if det["bbox"] is None:
            ax_pat.set_title(f"{nombre}: SIN DETECTAR", color="red", fontsize=11)
            print(f"{nombre:10s} | {'NO':8s} | {'-':7s} | -")
            continue

        n_detectadas += 1
        seg = segmentar_caracteres(det["gray"], det["bbox"])
        n_chars = len(seg["chars"])
        n_seis += n_chars == 6

        # Patente enderezada con los recuadros de cada carácter
        ax_pat.imshow(seg["roi_rot"], cmap="gray")
        for cx, cy, cw, ch, _ in seg["chars"]:
            ax_pat.add_patch(plt.Rectangle((cx, cy), cw, ch, fill=False,
                                           edgecolor="lime", linewidth=1.5))
        color = "black" if n_chars == 6 else "red"
        ax_pat.set_title(f"{nombre}: {n_chars} caracteres", color=color, fontsize=11)

        # Caracteres segmentados
        ax_car.imshow(tira_caracteres(seg), cmap="gray")

        metodo = seg.get("metodo", "")
        print(f"{nombre:10s} | {'SI':8s} | {n_chars:7d} | {seg['angulo']:+5.1f}° ({metodo})")

    fig.suptitle(f"Problema 2 - Patentes detectadas: {n_detectadas}/{len(archivos)} | "
                 f"con 6 caracteres segmentados: {n_seis}/{len(archivos)}", fontsize=14)
    plt.tight_layout()
    salida = os.path.join(CARPETA_SALIDA, "resumen_patentes_y_caracteres.png")
    plt.savefig(salida, dpi=120)
    plt.close(fig)

    print("-" * 50)
    print(f"Patentes detectadas: {n_detectadas}/{len(archivos)}")
    print(f"Con 6 caracteres segmentados: {n_seis}/{len(archivos)}")
    print(f"\nGuardado en {salida}")