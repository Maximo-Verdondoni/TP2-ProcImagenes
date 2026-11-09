# =============================================================================
# PROBLEMA 1 - PASO 4: REFINAMIENTO DE LAS MÁSCARAS (Punto A, etapa final)
# -----------------------------------------------------------------------------
# Objetivo: la máscara del paso 3 tiene dos defectos:
#   - monedas: algunas quedaron mordidas o con halo (ej. la moneda 17)
#   - dados: la máscara incluye la sombra y las caras laterales
# Este paso (a) separa monedas de dados por forma y (b) refina cada tipo con
# una técnica distinta. La lógica vive en segmentacion.py (refinar_objetos);
# este script solo la ejecuta, imprime resultados y arma la figura.
# =============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segmentacion import cargar_imagen, segmentar, refinar_objetos

# --- 1. Segmentación (pasos 1-3) y refinamiento (este paso) ---
img_bgr, small = cargar_imagen()                      # imagen a 1/4 (968x670)
small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)    # solo para mostrar con matplotlib
seg = segmentar(small)             # pipeline del paso 3: distancia Lab, histéresis, morfología, componentes
objetos = refinar_objetos(seg)     # clasifica cada componente (moneda/dado) y refina su máscara
L = seg["lab"][:, :, 0]            # canal de luminancia (se usa en el histograma de los dados)

monedas = [o for o in objetos if o["tipo"] == "moneda"]
dados = [o for o in objetos if o["tipo"] == "dado"]
print(f"Objetos: {len(objetos)} | monedas: {len(monedas)} (esperado 17) | dados: {len(dados)} (esperado 2)\n")

# Tabla de monedas: comparamos el área ANTES (máscara del paso 3, con halo o mordiscos)
# y DESPUÉS (círculo de Hough). 'método' dice qué técnica se usó para cada una.
print("MONEDAS")
print("id | método | circ  | área orig | área final | r final")
for o in sorted(monedas, key=lambda o: o["area"]):
    print(f"{o['id']:2d} | {o['metodo']:6s} | {o['circ0']:.2f}  | {o['area0']:9d} | {o['area']:10d} | {o['r']:.1f}")
# Si Hough no encuentra un círculo coherente, el código cae en la envolvente convexa
# como respaldo. Este contador confirma que en esta imagen nunca hizo falta (debe dar 0).
print(f"Monedas con respaldo hull: {sum(o['metodo'] == 'hull' for o in monedas)}")

# Tabla de dados: la máscara se achica en etapas (original -> silueta sin sombra -> cara superior)
print("\nDADOS")
for d in dados:
    print(f"Dado id {d['id']}: Otsu1 = {d['t1']:.0f} | Otsu2 = {d['t2']:.0f} | área orig {d['area0']} "
          f"-> silueta {cv2.countNonZero(d['mask_t1'])} -> cara antes del hull {cv2.countNonZero(d['nucleo'])} "
          f"-> cara superior final {d['area']}")

# --- 2. Helpers de visualización ---
def recorte(o, margen=15):
    # Ventana alrededor del bounding box del objeto, con 15 px de margen,
    # recortada para no salirse de la imagen (max evita índices negativos)
    x, y, w, h = o["bbox"]
    return (slice(max(y - margen, 0), y + h + margen),
            slice(max(x - margen, 0), x + w + margen))

def mostrar_zoom(ax, o, capas, titulo):
    """Muestra un recorte del objeto con los contornos de las máscaras superpuestos.
    capas: lista de (máscara, color)."""
    ys, xs = recorte(o)
    ax.imshow(small_rgb[ys, xs])
    for m, color in capas:
        ax.contour(m[ys, xs] > 0, levels=[0.5], colors=color, linewidths=1.3)
    ax.set_title(titulo, fontsize=10)
    ax.axis("off")

# --- 3. Figura ---
fig, ax = plt.subplots(2, 3, figsize=(18, 10))

# (0,0) Resultado general: cada objeto con su contorno refinado e id
ax[0, 0].imshow(small_rgb)
for o in objetos:
    color = "lime" if o["tipo"] == "moneda" else "red"
    ax[0, 0].contour(o["mask"] > 0, levels=[0.5], colors=color, linewidths=1.0)
    ax[0, 0].text(o["centro"][0], o["centro"][1], str(o["id"]), color="yellow",
                  fontsize=9, ha="center", va="center")
ax[0, 0].set_title("Verde = moneda (círculo), rojo = dado (cara superior)")
ax[0, 0].axis("off")

# (0,1) Máscara refinada completa: unión de todas las máscaras individuales
mask_ref = np.zeros(seg["mask_open"].shape, np.uint8)
for o in objetos:
    mask_ref[o["mask"] > 0] = 255
ax[0, 1].imshow(mask_ref, cmap="gray")
ax[0, 1].set_title("Máscara refinada")
ax[0, 1].axis("off")

# (0,2) Caso más difícil: la moneda de menor circularidad original (la 17, con bajo
# contraste, tenía el borde mordido). Rojo = máscara del paso 3, verde = círculo de Hough.
peor = min(monedas, key=lambda o: o["circ0"])
mostrar_zoom(ax[0, 2], peor, [(peor["mask_orig"], "red"), (peor["mask"], "lime")],
             f"Moneda {peor['id']} ({peor['metodo']}): rojo = antes ({peor['area0']}), verde = final ({peor['area']})")

# (1,0) y (1,1) Dados: celeste = silueta del dado completo (sin sombra), verde = solo la cara superior
for k, d in enumerate(dados[:2]):
    mostrar_zoom(
        ax[1, k],
        d,
        [(d["mask_t1"], "cyan"), (d["mask"], "lime")],
        f"Dado {d['id']}: celeste = dado entero, verde = cara superior",
    )

# (1,2) Histograma de L (luminancia) dentro del primer dado. Muestra por qué hacen
# falta DOS umbrales de Otsu: hay tres grupos de intensidad (ver conclusión).
if dados:
    d0 = dados[0]
    ax[1, 2].hist(L[d0["mask_orig"] > 0].ravel(), bins=64)
    ax[1, 2].axvline(d0["t1"], color="orange", label=f"Otsu1 = {d0['t1']:.0f}")
    ax[1, 2].axvline(d0["t2"], color="r", label=f"Otsu2 = {d0['t2']:.0f}")
    ax[1, 2].legend()
    ax[1, 2].set_title(f"Dado {d0['id']}: histograma de L")

plt.tight_layout()
plt.savefig("ej1/paso4_resultado.png", dpi=100)
print("\nFigura guardada en ej1/paso4_resultado.png")
plt.show()  

# QUÉ HACE refinar_objetos (segmentacion.py), para poder explicarlo:
#
# 1) Separar monedas de dados por CIRCULARIDAD (Unidad 7.2):
#    C = 4*pi*Área / Perímetro^2, calculada sobre la máscara del paso 3.
#    En la salida del paso 3 las monedas dan 0.88-0.92 y los dados 0.81-0.82;
#    el corte es 0.85. Ojo: el margen entre grupos es angosto (0.82 vs 0.88).
#    Se sostiene porque el pipeline valida después: Hough tiene que encontrar un
#    círculo en las "monedas" y los dados tienen que tener puntos redondos.
#
# 2) MONEDAS -> Transformada de Hough circular (Unidad 3.4) dentro de un ROI
#    alrededor de cada moneda. El radio buscado se acota a partir del área de la
#    máscara (0.85 a 1.3 veces el radio equivalente), y se acepta solo un círculo
#    cuyo centro esté cerca del centro de la máscara. Se usa el borde REAL de la
#    moneda, no la máscara con halo o mordiscos: por eso la 17 queda bien.
#    La envolvente convexa (6.4) quedó solo como respaldo.
#
# 3) DADOS -> tres cosas combinadas:
#    a) 1er Otsu sobre L dentro del objeto: separa el dado de su sombra. La
#       silueta es la máscara rellena de lo claro.
#    b) 2do Otsu sobre lo claro: la cara superior es la MÁS clara (histograma
#       de la figura: sombra ~50, caras laterales ~150-190, cara superior ~215-225).
#    c) Los puntos negros REDONDOS (circularidad > 0.72, área > 25 px) marcan la cara
#       superior, porque los de las caras laterales salen elípticos por la
#       perspectiva. Se dilatan (11x11) y se recortan con la silueta; eso recupera
#       los bordes de las esquinas donde hay un punto. El resultado es la unión
#       de la cara blanca con esas esquinas, y al final una envolvente convexa
#       tapa los puntos negros.