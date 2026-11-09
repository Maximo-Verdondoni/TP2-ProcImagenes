import cv2, os
import matplotlib.pyplot as plt
from paso1 import CARPETA
from paso2 import segmentar_caracteres

img = cv2.imread(os.path.join(CARPETA, "img04.png"))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

for bbox in [(287, 172, 128, 59), (287, 171, 128, 70)]:
    seg = segmentar_caracteres(gray, bbox)
    print(f"{bbox}: ángulo={seg['angulo']:+.1f}° ({seg['metodo']}) | caracteres={len(seg['chars'])}")

    plt.figure(figsize=(8, 3))
    plt.imshow(seg["roi_rot"], cmap="gray")
    for cx, cy, cw, ch, _ in seg["chars"]:
        plt.gca().add_patch(plt.Rectangle((cx, cy), cw, ch, fill=False, edgecolor="lime"))
    plt.title(f"{bbox}: {len(seg['chars'])} caracteres"); plt.axis("off")
    plt.savefig(f"test_img04_{bbox[3]}.png", dpi=110); plt.close()