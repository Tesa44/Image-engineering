from PIL import Image
from matplotlib import pyplot as plt

def bresenham_line(x0, y0, x1, y1):
    """Zwraca listę punktów tworzących linię za pomocą algorytmu Bresenhama"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

def fill_triangle(image, vertices, color):
    """Wypełnia trójkąt z zadanych wierzchołków"""
    edges = []
    for i in range(3):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % 3]
        edges.extend(bresenham_line(p1[0], p1[1], p2[0], p2[1]))

    # Grupa punktów według y (linie poziome)
    scanlines = {}
    for x, y in edges:
        if y not in scanlines:
            scanlines[y] = []
        scanlines[y].append(x)

    for y in scanlines:
        x_vals = sorted(scanlines[y])
        if len(x_vals) >= 2:
            for x in range(x_vals[0], x_vals[-1] + 1):
                if 0 <= x < 50 and 0 <= y < 50:
                    image.putpixel((x, y), color)

def draw_line(image,line, color=(0,255,0)):
    p1 = line[0]
    p2 = line[1]
    points = bresenham_line(p1[0], p1[1], p2[0], p2[1])
    for point in points:
        image.putpixel(point, color)


# Tworzymy mały obraz 50x50
img = Image.new("RGB", (50, 50), "white")

# Wierzchołki trójkąta w przestrzeni 50x50
triangle = [(10, 10), (40, 15), (20, 40)]
line = [(10,30), (40,40)]
triangle2 = [(0,50),(10,50),(5,40)]

fill_triangle(img, triangle, color=(255, 0, 0))
draw_line(img, line, color=(0, 255, 0))
fill_triangle(img, triangle2, color=(0, 0, 255))
plt.imshow(img)
plt.show()

img.save("triangle_filled.png")

