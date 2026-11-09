# =============================================================================
# PROBLEMA 1 - PASO 2: PRIMER INTENTO DE BINARIZACIÓN (INTENTO DESCARTADO)
# -----------------------------------------------------------------------------
# Objetivo: con el fondo ya estimado (paso 1), separar objetos de fondo.
# Se prueban DOS señales independientes y se combinan con una unión:
#   (a) |gris - fondo| con Otsu   -> detecta objetos que difieren en luminancia
#   (b) canal S (saturación) con Otsu -> detecta objetos con color (dorados)
# Resultado: la combinación es frágil (ver CONCLUSIÓN al final). Este fracaso
# motiva el paso 3, donde se usa una única medida de color (distancia en CIELAB)
# y un umbral con histéresis.
# =============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Carga (Unidad 1.3) ---
img_bgr = cv2.imread("ej1/monedas.jpg")
assert img_bgr is not None, "No se encontró monedas.jpg"
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)     # solo para mostrar con matplotlib
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# --- 2. Fondo por mediana (resultado del paso 1) ---
def estimar_fondo(g, f=8, k=101, sigma=15):
    # Reducimos a 1/f por velocidad: el fondo es una superficie suave, no
    # necesita resolución completa. INTER_AREA evita aliasing al reducir.
    small = cv2.resize(g, None, fx=1/f, fy=1/f, interpolation=cv2.INTER_AREA)
    # Mediana con ventana de 101x101 a esta escala: las monedas (32-46 px) ocupan
    # una fracción chica de la ventana, así que la mediana devuelve el valor del
    # fondo (filtro no lineal, Unidad 2.4).
    fondo = cv2.medianBlur(small, k)
    # Un Gaussiano suaviza los escalones que deja la mediana.
    fondo = cv2.GaussianBlur(fondo, (0, 0), sigma)
    # Volvemos al tamaño original (el fondo es suave, la bilineal alcanza).
    return cv2.resize(fondo, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)

fondo = estimar_fondo(gray)

# --- 3. Señal (a): diferencia absoluta respecto del fondo, en gris ---
# Se resta en float32 (en uint8 la resta desborda) y se toma el VALOR ABSOLUTO
# para detectar tanto objetos más claros que el fondo (dados, plateadas) como
# más oscuros (en teoría, las doradas). convertScaleAbs devuelve uint8.
absdiff = cv2.convertScaleAbs(gray.astype(np.float32) - fondo.astype(np.float32))
# Suavizado Gaussiano (Unidad 2.4): atenúa el grabado interno de las monedas
# y el ruido, para que cada moneda se comporte como una región más homogénea.
absdiff = cv2.GaussianBlur(absdiff, (0, 0), 3)

# Otsu (Unidad 3.1): elige el umbral que maximiza la varianza entre clases.
# Supone un histograma razonablemente bimodal; ver la limitación en la conclusión.
t_gray, mask_gray = cv2.threshold(absdiff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu sobre |diff|: umbral = {t_gray:.1f} | foreground = {100*np.mean(mask_gray>0):.1f}%")

# --- 4. Señal (b): canal S (saturación) de HSV (Unidad 5.4) ---
# El fondo es gris (saturación casi nula), mientras que las monedas doradas y el
# centro de las bimetálicas tienen saturación alta. Es una señal que no depende
# de la iluminación de izquierda a derecha tanto como el gris. Limitación: con
# brillo (V) bajo la saturación se vuelve inestable y ruidosa, lo que explica las
# manchas en el fondo oscuro de la izquierda (ver punto 2 de la conclusión).
hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
S = cv2.GaussianBlur(hsv[:, :, 1], (0, 0), 3)
t_s, mask_s = cv2.threshold(S, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu sobre S: umbral = {t_s:.1f} | foreground = {100*np.mean(mask_s>0):.1f}%")

# --- 5. Unión de ambas máscaras (OR lógico) ---
# La idea: lo que una señal pierde, lo recupera la otra.
mask_union = cv2.bitwise_or(mask_gray, mask_s)
print(f"Unión: foreground = {100*np.mean(mask_union>0):.1f}%")

# --- 6. Visualización ---
fig, ax = plt.subplots(2, 4, figsize=(20, 9))
ax[0, 0].imshow(img_rgb);                  ax[0, 0].set_title("Original")
ax[0, 1].imshow(absdiff, cmap="gray");     ax[0, 1].set_title("|gris - fondo| (suavizado)")
ax[0, 2].imshow(mask_gray, cmap="gray");   ax[0, 2].set_title(f"Máscara |diff| (Otsu={t_gray:.0f})")
ax[0, 3].hist(absdiff.ravel(), bins=128, log=True)     # escala log: el pico del fondo domina
ax[0, 3].axvline(t_gray, color="r");       ax[0, 3].set_title("Histograma |diff| (log) + Otsu")
ax[1, 0].imshow(S, cmap="gray");           ax[1, 0].set_title("Canal S (suavizado)")
ax[1, 1].imshow(mask_s, cmap="gray");      ax[1, 1].set_title(f"Máscara S (Otsu={t_s:.0f})")
ax[1, 2].imshow(mask_union, cmap="gray");  ax[1, 2].set_title("Unión |diff| ∪ S")
ax[1, 3].imshow(img_rgb)
ax[1, 3].contour(mask_union > 0, levels=[0.5], colors="lime", linewidths=0.8)
ax[1, 3].set_title("Contorno de la unión sobre el original")
for a in ax.ravel():
    if a is not ax[0, 3]:          # el histograma conserva sus ejes
        a.axis("off")
plt.tight_layout()
plt.savefig("ej1/paso2_resultado.png", dpi=100)
plt.show()   # sin tkinter solo da un warning; el resultado queda en el PNG

# CONCLUSIÓN DE ESTE PASO (y por qué se descarta):
#  1) |gris - fondo| con Otsu = 50 queda demasiado alto. El histograma está
#     dominado por el pico del fondo (cerca de 0) y los objetos son una cola
#     chica (~14% de la imagen), de modo que Otsu, que asume dos clases
#     comparables, corta en el lugar equivocado. Solo sobreviven los objetos
#     más brillantes (dados, plateadas de la izquierda) y contornos rotos en
#     las plateadas del lado derecho, donde el fondo es claro y el contraste baja.
#  2) El canal S (Otsu = 66) rellena bien las doradas y el centro de las
#     bimetálicas, pero pierde las plateadas, los anillos de las bimetálicas
#     y los dados, que casi no tienen saturación, y deja manchas de ruido en el
#     fondo oscuro de la izquierda.
#  3) La unión resulta un parche: cada señal tiene su propio Otsu mal ubicado
#     y sus propios agujeros. No es robusta.
#  -> Lo que se aprendió y se aplica en el paso 3:
#       - una única medida que combine luminancia y color: distancia en CIELAB
#         respecto del fondo (Unidad 5.4), que cubre doradas, plateadas y dados;
#       - un umbral que no dependa de Otsu: histéresis con dos umbrales fijos
#         tomados del valle del histograma, que completa los bordes débiles;
#       - morfología (Unidad 6) para completar las máscaras, que aquí no se aplicó.