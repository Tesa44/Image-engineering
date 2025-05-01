import matplotlib.pyplot as plt
from PIL import Image

def interpolate_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def bresenham_line_with_color(x0, y0, c0, x1, y1, c1, image):
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
    return (p[0] - v0[0]) * (v1[1] - v0[1]) - (p[1] - v0[1]) * (v1[0] - v0[0])

def draw_filled_triangle_with_color(image, v0, v1, v2):
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

            if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                alpha = w0 / area
                beta = w1 / area
                gamma = w2 / area

                color = (
                    int(c0[0] * alpha + c1[0] * beta + c2[0] * gamma),
                    int(c0[1] * alpha + c1[1] * beta + c2[1] * gamma),
                    int(c0[2] * alpha + c1[2] * beta + c2[2] * gamma),
                )
                image.putpixel((x, y), color)

def downscale_image(img, scale):
    w_out = img.width // scale
    h_out = img.height // scale
    downscaled = Image.new("RGB", (w_out, h_out))

    for y in range(h_out):
        for x in range(w_out):
            r = g = b = 0
            for dy in range(scale):
                for dx in range(scale):
                    px = img.getpixel((x * scale + dx, y * scale + dy))
                    r += px[0]
                    g += px[1]
                    b += px[2]
            area = scale * scale
            avg_color = (r // area, g // area, b // area)
            downscaled.putpixel((x, y), avg_color)

    return downscaled


def main():
    scale = 2
    width, height = 50, 50
    img_scaled = Image.new("RGB", (width * scale, height * scale), "white")

    # # Przykładowa linia (skalowane współrzędne)
    # bresenham_line_with_color(
    #     5 * scale, 5 * scale, (255, 0, 0),
    #     45 * scale, 10 * scale, (0, 0, 255),
    #     img_scaled
    # )

    # Przykładowy trójkąt z interpolacją koloru (skalowane współrzędne)
    v0 = (10 * scale, 10 * scale, (255, 0, 0))
    v1 = (40 * scale, 15 * scale, (0, 255, 0))
    v2 = (20 * scale, 40 * scale, (0, 0, 255))
    draw_filled_triangle_with_color(img_scaled, v0, v1, v2)

    # Skalowanie z powrotem do 50x50 z uśrednianiem
    # img_result = img_scaled.resize((width, height), Image.Resampling.BOX)
    img_result = downscale_image(img_scaled, scale)
    plt.imshow(img_result)
    plt.show()
    # img_result.save("triangle_antialiased.png")
    # img_result.show()

if __name__ == "__main__":
    main()
