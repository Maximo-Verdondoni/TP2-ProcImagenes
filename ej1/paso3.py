# =============================================================================
# PROBLEMA 1 - PASO 3: SEGMENTACIÓN FINAL DE OBJETOS (Punto A)
# -----------------------------------------------------------------------------
# Objetivo: obtener una máscara limpia donde cada moneda y cada dado sea un
# componente conectado separado (esperado: 19 = 17 monedas + 2 dados).
# Pipeline:
#   1) distancia de color al fondo en CIELAB (Unidad 5.4)
#   2) umbral con histéresis, o sea reconstrucción geodésica acotada (Unidades 3.1 y 6.5)
#   3) morfología: clausura, relleno de huecos, apertura (Unidad 6)
#   4) componentes conectados y filtro por área (Unidad 3.3)
# Esta versión muestra cada etapa. La misma lógica está refactorizada en
# segmentacion.py (función segmentar), que reutilizan los pasos 4 y 5: si se
# cambia un parámetro acá, hay que cambiarlo allá también.
# =============================================================================
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Carga y reducción a 1/4 ---
# Trabajamos a 1/4 (968x670) por velocidad: a esta escala las monedas miden
# 64-92 px de diámetro, de sobra para segmentarlas y medir formas.
# INTER_AREA promedia píxeles al reducir y evita aliasing (Unidad 2.1).
img_bgr = cv2.imread("ej1/monedas.jpg")
assert img_bgr is not None, "No se encontró monedas.jpg"
small = cv2.resize(img_bgr, None, fx=1/4, fy=1/4, interpolation=cv2.INTER_AREA)
small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)     # solo para mostrar con matplotlib

# Pasamos a CIELAB (Unidad 5.4): separa la luminancia (L) de la información de
# color (a*: verde-rojo, b*: azul-amarillo). Es la clave de este paso: el gris
# solo (paso 2) confunde las monedas doradas con el fondo porque tienen luminancia
# parecida, pero se diferencian claramente en b* (amarillo).
# Nota: OpenCV en 8 bits escala L a 0-255 y desplaza a*, b* en +128, así que las
# distancias que siguen están en "unidades OpenCV", no exactamente en ΔE de la CIE.
lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
print("Imagen de trabajo:", small.shape)

# --- 2. Estimación del fondo, un canal por vez (mismo método que el paso 1) ---
def fondo_canal(c, k=101, sigma=15):
    # Se reduce otra vez a la mitad (queda 1/8 del original, igual que en el
    # paso 1): a esa escala las monedas miden 32-46 px y la ventana de 101x101
    # de la mediana las ignora (cada una ocupa una fracción chica de la ventana).
    tiny = cv2.resize(c, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    f = cv2.medianBlur(tiny, k)                 # filtro no lineal (Unidad 2.4); k impar
    f = cv2.GaussianBlur(f, (0, 0), sigma)      # suaviza los escalones de la mediana
    return cv2.resize(f, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_LINEAR)

# El fondo se estima por separado en L, a* y b* y se apilan en una imagen de 3 canales
fondo_lab = np.dstack([fondo_canal(lab[:, :, i]) for i in range(3)])

# --- 3. Distancia de color al fondo ---
# Se resta en float32 (en uint8 desbordaría) y se calcula la norma euclídea en
# el espacio Lab: un solo número por píxel que resume "cuánto se aleja este píxel
# del fondo en luminancia Y en color". Cubre doradas (difieren en b*),
# plateadas y dados (difieren en L) con una única señal.
diff = lab.astype(np.float32) - fondo_lab.astype(np.float32)
dist = np.sqrt((diff ** 2).sum(axis=2))
# Suavizado leve (sigma = 2) para atenuar el ruido y el grabado de las monedas
dist = cv2.GaussianBlur(dist, (0, 0), 2)

# --- 4. Umbral con histéresis (reconstrucción geodésica acotada, Unidad 6.5) ---
# Por qué no Otsu: en el paso 2 falló porque el fondo domina el histograma.
# Acá los dos umbrales se eligen mirando el histograma de la figura:
#   - el fondo es un pico cerca de 0-10
#   - los objetos forman una joroba desde ~30
#   T_ALTO = 25: cae en el valle; lo que lo supera es objeto con seguridad (semillas).
#   T_BAJO = 14: más permisivo; sirve para completar bordes débiles (monedas
#                plateadas de bajo contraste), pero solo si están conectados a una semilla.
T_ALTO, T_BAJO, N_GROW = 25, 14, 10
print(f"Umbrales: alto={T_ALTO} | bajo={T_BAJO} | iteraciones de crecimiento={N_GROW}")

seeds = ((dist > T_ALTO) * 255).astype(np.uint8)    # núcleo seguro de cada objeto
low = ((dist > T_BAJO) * 255).astype(np.uint8)      # zona candidata (incluye la anterior)
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
mask0 = seeds.copy()
# Crecimiento: dilatar las semillas y quedarse solo con lo que cae dentro de "low".
# Es una dilatación geodésica de N pasos (Unidad 6.5). Si se iterara hasta que la
# máscara no cambie más, sería la reconstrucción morfológica completa (histéresis
# clásica); acá se corta en N_GROW pasos a propósito:
# cada iteración agrega como máximo 1 px, así que el máscara crece a lo sumo 10 px
# desde las semillas. Ese límite evita que el halo de sombra conecte objetos
# vecinos. Como las semillas están contenidas en "low", nunca se pierden.
for _ in range(N_GROW):
    mask0 = cv2.dilate(mask0, k3) & low
print(f"Foreground: semillas {100 * np.mean(seeds > 0):.1f}% -> tras crecer {100 * np.mean(mask0 > 0):.1f}%")

# --- 5. Morfología (Unidad 6) ---
def rellenar_huecos(m):
    # Relleno de huecos por floodFill desde el fondo: se inunda el fondo desde la
    # esquina (0,0) y todo lo que NO quedó inundado es un hueco interno del objeto.
    # Supone que (0,0) es fondo, de ahí el assert.
    assert m[0, 0] == 0, "La esquina (0,0) no es fondo"
    ff = m.copy()
    pad = np.zeros((m.shape[0] + 2, m.shape[1] + 2), np.uint8)   # floodFill exige una máscara 2 px más grande
    cv2.floodFill(ff, pad, (0, 0), 255)
    # not(ff) vale 255 solo en los huecos; el OR con la máscara original los rellena
    return m | cv2.bitwise_not(ff)

# 5.1 Clausura (dilatación + erosión) 7x7: sella fisuras finas en los contornos
k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask_close = cv2.morphologyEx(mask0, cv2.MORPH_CLOSE, k_close)

# 5.2 Relleno de huecos: rellena los huecos que dejan los relieves de las monedas
# y los puntos de los dados, para obtener máscaras sólidas
mask_fill = rellenar_huecos(mask_close)

# 5.3 Apertura (erosión + dilatación) con elemento elíptico de 31x31:
#  - elimina estructuras más chicas que 31 px (restos de sombra, ruido)
#  - corta puentes finos entre objetos pegados
# Es seguro porque las monedas miden 64-92 px, bastante más que el elemento.
k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
mask_open = cv2.morphologyEx(mask_fill, cv2.MORPH_OPEN, k_open)

# --- 6. Componentes conectados y descriptores (Unidades 3.3 y 7.2) ---
# connectivity=8: dos píxeles en diagonal también se consideran conectados
n, labels, stats, cents = cv2.connectedComponentsWithStats(mask_open, connectivity=8)
# La moneda más chica mide ~3200 px de área a esta escala; 1000 descarta ruido sin riesgo
AREA_MIN = 1000
keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= AREA_MIN]   # el 0 es el fondo
print(f"\nComponentes con área >= {AREA_MIN}: {len(keep)} (esperado: 19 = 17 monedas + 2 dados)")
for i in sorted(keep, key=lambda j: stats[j, cv2.CC_STAT_AREA]):
    x, y, ww, hh, a = stats[i]
    cnts, _ = cv2.findContours((labels == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    per = cv2.arcLength(cnts[0], True)
    # Circularidad (factor de forma) = 4*pi*Área / Perímetro^2: vale 1 para un círculo
    # perfecto y baja con formas angulosas. En la salida las monedas dan 0.88-0.92 y
    # los dados 0.81-0.82; ese descriptor separa ambos en el paso 4.
    circ = 4 * np.pi * a / per ** 2
    print(f"id {i:2d} | área {a:5d} | bbox {ww}x{hh} | circ {circ:.2f} | centro ({cents[i][0]:.0f}, {cents[i][1]:.0f})")

# Máscara final: solo los componentes que pasaron el filtro de área
mask_final = (np.isin(labels, keep) * 255).astype(np.uint8)
cv2.imwrite("ej1/mascara_paso3.png", mask_final)   # salida de referencia (los pasos siguientes no la leen)

# --- 7. Visualización de todas las etapas ---
fig, ax = plt.subplots(2, 4, figsize=(20, 9))
ax[0, 0].imshow(small_rgb);                 ax[0, 0].set_title("Original (1/4)")
ax[0, 1].imshow(dist, cmap="magma");        ax[0, 1].set_title("Distancia Lab al fondo")
ax[0, 2].hist(dist.ravel(), bins=128, log=True)     # escala log: el pico del fondo domina
ax[0, 2].axvline(T_BAJO, color="orange")
ax[0, 2].axvline(T_ALTO, color="r");        ax[0, 2].set_title(f"Histograma (log) + umbrales {T_BAJO} / {T_ALTO}")
ax[0, 3].imshow(mask0, cmap="gray");        ax[0, 3].set_title(f"Histéresis ({T_ALTO} / {T_BAJO})")
ax[1, 0].imshow(mask_close, cmap="gray");   ax[1, 0].set_title("Clausura 7x7")
ax[1, 1].imshow(mask_fill, cmap="gray");    ax[1, 1].set_title("Relleno de huecos")
ax[1, 2].imshow(mask_open, cmap="gray");    ax[1, 2].set_title("Apertura 31x31")
ax[1, 3].imshow(small_rgb)
ax[1, 3].contour(mask_final > 0, levels=[0.5], colors="lime", linewidths=0.8)
for i in keep:
    ax[1, 3].text(cents[i][0], cents[i][1], str(i), color="yellow", fontsize=9,
                  ha="center", va="center")
ax[1, 3].set_title(f"Resultado: {len(keep)} objetos")
for a in ax.ravel():
    if a is not ax[0, 2]:          # el histograma conserva sus ejes
        a.axis("off")
plt.tight_layout()
plt.savefig("ej1/paso3_resultado.png", dpi=100)
plt.show()   # sin tkinter solo da un warning; el resultado queda en el PNG

# HISTORIA DE AJUSTES (para justificar los parámetros):
#  - Versión 1: umbral único bajo (16) + clausura 15x15 -> la clausura grande
#    puenteó objetos vecinos (3 pares quedaron fusionados: 16 objetos en vez de 19).
#  - Versión 2: umbral único 25 + clausura chica + apertura 31x31 -> separó todo
#    pero perdió una moneda plateada de bajo contraste (18 objetos) y dejó
#    mordiscos en los bordes de otras.
#  - Versión final: histéresis 25/14 con crecimiento acotado + clausura 7x7 ->
#    19 objetos, monedas completas y sin fusiones.