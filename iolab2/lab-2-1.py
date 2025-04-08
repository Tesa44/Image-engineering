import matplotlib.pyplot as plt
import numpy as np
import cv2
from dateutil.utils import within_delta


def save_ppm_p6(filename, image):
    """Zapisuje obraz w formacie P6 (binarny PPM)."""
    height, width, _ = image.shape
    with open(filename, 'wb') as f:
        f.write(f"P6\n{width} {height}\n255\n".encode())
        image.tofile(f)

def generate_gradient(colors, width=600, height=50):
    """Generuje gradient przejścia między podanymi kolorami."""
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    num_colors = len(colors)

    for i in range(width):
        t = i / (width - 1)
        idx = int(t * (num_colors - 1))
        next_idx = min(idx + 1, num_colors - 1)
        local_t = (t * (num_colors - 1)) - idx
        color = tuple(int((1 - local_t) * colors[idx][j] + local_t * colors[next_idx][j]) for j in range(3))
        gradient[:, i] = color

    return gradient


def draw_rgb_spectrum():
    """Rysuje dwa spektra przejść kolorów RGB."""
    # image = np.ones((120, 600, 3), dtype=np.uint8) * 255  # Białe tło
    image = np.zeros((100,1792,3),dtype=np.uint8)
    width = 1792
    num_colors = 8
    colors = [(0, 0, 0), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 255, 0), (255, 0, 0), (255, 0, 255),
                  (255, 255, 255)]
    for i in range(width):
        t = i / (width - 1)
        idx = int(t * (num_colors - 1))
        next_idx = min(idx + 1, num_colors - 1)
        local_t = (t * (num_colors - 1)) - idx
        color = tuple(int((1 - local_t) * colors[idx][j] + local_t * colors[next_idx][j]) for j in range(3))
        image[:, i] = color
    return image


# Generowanie spektrum i zapisywanie w PPM (P6)
rgb_spectrum = draw_rgb_spectrum()
save_ppm_p6("rgb_spectrum.ppm", rgb_spectrum)

# Wyświetlenie obrazu
plt.imshow(rgb_spectrum)
plt.show()
# cv2.imshow("RGB Spectrum", rgb_spectrum)
# cv2.waitKey(0)
# cv2.destroyAllWindows()