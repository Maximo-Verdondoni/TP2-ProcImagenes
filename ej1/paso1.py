# =============================================================================
# PROBLEMA 1 - PASO 1: ANÁLISIS DE LA ILUMINACIÓN Y ESTIMACIÓN DEL FONDO
# -----------------------------------------------------------------------------
# Objetivo: el enunciado dice que el fondo tiene intensidad NO uniforme.
# Antes de segmentar hay que (1) medir cuánto varía la iluminación y
# (2) probar cómo estimar ese fondo para poder compensarlo.
# Este paso es exploratorio: sirve para justificar las decisiones del pipeline
# final (en el pipeline final se usa mediana, no apertura; ver más abajo).
# =============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt


# --- 1. Carga (Unidad 1.3: lectura, conversiones, visualización) ---
img_bgr = cv2.imread("ej1/monedas.jpg")          # OpenCV lee en orden BGR, no RGB
assert img_bgr is not None, "No se encontró monedas.jpg"
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)    # matplotlib espera RGB
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)      # luminancia: 0.299 R + 0.587 G + 0.114 B

# Resultado esperado: (2681, 3871, 3) uint8 | gris: min 0, max 255, media ~107
print("Shape:", img_bgr.shape, "| dtype:", img_bgr.dtype)
print("Gris -> min:", gray.min(), "max:", gray.max(), "media:", round(gray.mean(), 1))

# --- 2. Perfil del gradiente de iluminación ---
# Media de intensidad por columna y por fila. Cuantifica el problema:
# por columna sube de ~50 (izquierda) a ~175 (derecha), más de 3 veces, por eso
# un umbral global sobre el gris no sirve. Por fila varía mucho menos (~90 a ~125).
perfil_cols = gray.mean(axis=0)     # un valor por columna
perfil_filas = gray.mean(axis=1)    # un valor por fila

# --- 3. Estimación del fondo ---
# Se trabaja en escala 1/8 por velocidad: un filtro de mediana con kernel
# grande sobre la imagen completa (3871x2681) sería muy lento. El fondo es
# una superficie suave, así que perder resolución no importa.
# INTER_AREA promedia los píxeles que se juntan (evita aliasing al reducir).
f = 8
small = cv2.resize(gray, None, fx=1/f, fy=1/f, interpolation=cv2.INTER_AREA)

# Opción A: filtro de MEDIANA con kernel grande (filtro no lineal, Unidad 2.4).
# La mediana ignora los objetos que ocupan menos de la mitad de la ventana.
# A esta escala las monedas miden 32-46 px de diámetro y la ventana es de
# 101x101, así que una moneda ocupa como mucho ~16% de la ventana y la mediana
# devuelve el valor del fondo. Sirve igual para objetos más claros (dados,
# monedas plateadas) que para más oscuros (monedas doradas).
fondo_med = cv2.medianBlur(small, 101)               # el kernel debe ser impar
# La mediana deja escalones; un Gaussiano (sigma = 15 px a esta escala) los
# suaviza y deja una superficie continua, que es lo que esperamos del fondo.
fondo_med = cv2.GaussianBlur(fondo_med, (0, 0), 15)

# Opción B: APERTURA morfológica (Unidad 6.3) con elemento estructural elíptico.
# Apertura = erosión + dilatación. Elimina estructuras CLARAS más chicas que el
# elemento estructural. Usamos 81x81: casi el doble del diámetro de la moneda
# más grande (46 px a esta escala), así que ninguna moneda "entra" en el elemento.
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
fondo_open = cv2.morphologyEx(small, cv2.MORPH_OPEN, se)
fondo_open = cv2.GaussianBlur(fondo_open, (0, 0), 15)

# Volver a la resolución original (interpolación bilineal: el fondo es suave)
h, w = gray.shape
fondo_med = cv2.resize(fondo_med, (w, h), interpolation=cv2.INTER_LINEAR)
fondo_open = cv2.resize(fondo_open, (w, h), interpolation=cv2.INTER_LINEAR)

# --- 4. Corrección: imagen - fondo ---
def corregir(g, fondo):
    # Se pasa a float32 antes de restar: en uint8 la resta desborda (ej. 10 - 20 = 246).
    d = g.astype(np.float32) - fondo.astype(np.float32)
    # Esta diferencia tiene signo (objetos más claros que el fondo > 0, más oscuros < 0).
    # Se reescala a 0-255 solo para poder visualizarla.
    return cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

corr_med = corregir(gray, fondo_med)
corr_open = corregir(gray, fondo_open)

# --- 5. Visualización ---
fig, ax = plt.subplots(2, 4, figsize=(20, 9))
ax[0, 0].imshow(img_rgb);               ax[0, 0].set_title("Original")
ax[0, 1].imshow(gray, cmap="gray");     ax[0, 1].set_title("Gris")
ax[0, 2].plot(perfil_cols);             ax[0, 2].set_title("Media por columna (izq → der)")
ax[0, 3].plot(perfil_filas);            ax[0, 3].set_title("Media por fila (arriba → abajo)")
ax[1, 0].imshow(fondo_med, cmap="gray");  ax[1, 0].set_title("Fondo (mediana)")
ax[1, 1].imshow(fondo_open, cmap="gray"); ax[1, 1].set_title("Fondo (apertura)")
ax[1, 2].imshow(corr_med, cmap="gray");   ax[1, 2].set_title("Corregida (mediana)")
ax[1, 3].imshow(corr_open, cmap="gray");  ax[1, 3].set_title("Corregida (apertura)")
# Sin ejes en las imágenes (los gráficos de perfil conservan los suyos)
for a in ax.ravel()[:2].tolist() + ax[1].tolist():
    a.axis("off")

plt.tight_layout()
plt.savefig("ej1/paso1_resultado.png", dpi=100)
plt.show()   # sin tkinter solo da un warning; el resultado queda en el PNG

# CONCLUSIÓN DE ESTE PASO (justifica el pipeline final):
#  - La iluminación varía más de 3x de izquierda a derecha -> hay que compensar el fondo.
#  - Mediana: fondo suave y sin rastros de monedas -> se usa en el pipeline final.
#  - Apertura: deja una mancha oscura donde están las monedas y halos brillantes
#    alrededor. Es lógico: la apertura solo elimina estructuras CLARAS, y las
#    monedas doradas son más oscuras que el fondo del lado derecho, así que
#    quedan "dentro" del fondo estimado. -> descartada.
#  - Limitación: restar en gris no alcanza, porque las doradas tienen una
#    luminancia casi igual a la del fondo. Por eso el pipeline final mide la
#    diferencia en color (distancia en CIELAB), que sí las distingue.