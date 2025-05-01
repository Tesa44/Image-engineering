from PIL import Image
from matplotlib import pyplot as plt

def interpolate_color(c1, c2, t):
    """Interpoluje kolor między c1 a c2 w zakresie t ∈ [0,1]"""
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def bresenham_line_with_color(x0, y0, c0, x1, y1, c1, image):
    """Rysuje linię z interpolacją koloru"""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    steps = max(dx, dy)
    t = 0
    i = 0

    while True:
        if 0 <= x0 < image.width and 0 <= y0 < image.height:
            interp_color = interpolate_color(c0, c1, i / steps if steps > 0 else 0)
            image.putpixel((x0, y0), interp_color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
        i += 1

def edge_function(v0, v1, p):
    """Pomocnicza funkcja do barycentrycznych współczynników"""
    return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])

def draw_filled_triangle_with_color(image, v0, v1, v2):
    """Wypełnianie trójkąta z interpolacją koloru (barycentryczna interpolacja)"""
    # Wierzchołki i kolory
    x0, y0, c0 = v0
    x1, y1, c1 = v1
    x2, y2, c2 = v2

    min_x = max(min(x0, x1, x2), 0)
    max_x = min(max(x0, x1, x2), image.width - 1)
    min_y = max(min(y0, y1, y2), 0)
    max_y = min(max(y0, y1, y2), image.height - 1)

    area = edge_function((x0, y0), (x1, y1), (x2, y2))

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            w0 = edge_function((x1, y1), (x2, y2), (x, y))
            w1 = edge_function((x2, y2), (x0, y0), (x, y))
            w2 = edge_function((x0, y0), (x1, y1), (x, y))

            # if w0 >= 0 and w1 >= 0 and w2 >= 0:
            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                # Współczynniki barycentryczne
                alpha = w0 / area
                beta = w1 / area
                gamma = w2 / area

                color = (
                    int(c0[0] * alpha + c1[0] * beta + c2[0] * gamma),
                    int(c0[1] * alpha + c1[1] * beta + c2[1] * gamma),
                    int(c0[2] * alpha + c1[2] * beta + c2[2] * gamma),
                )
                image.putpixel((x, y), color)

img = Image.new("RGB", (50, 50), "white")

# Przykładowa linia z kolorami końcowych punktów
bresenham_line_with_color(5, 5, (255, 0, 0), 45, 10, (0, 0, 255), img)

# Przykładowy trójkąt z kolorowymi wierzchołkami
v0 = (10, 10, (255, 0, 0))    # czerwony
v1 = (40, 15, (0, 255, 0))    # zielony
v2 = (20, 40, (0, 0, 255))    # niebieski
draw_filled_triangle_with_color(img, v0, v1, v2)

plt.imshow(img)
plt.show()

# img.save("triangle_color_interpolated.png")


