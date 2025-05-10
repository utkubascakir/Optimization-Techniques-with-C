import os
import numpy as np
from skimage.draw import line  


def generate_detailed_circle_image(N, num_circles=5, add_patterns=True):
    image = np.random.rand(N, N) * 0.5 
    for _ in range(num_circles):
        center_x, center_y = np.random.randint(N), np.random.randint(N)
        radius = np.random.randint(N // 10, N // 3)
        for i in range(N):
            for j in range(N):
                if (i - center_x) ** 2 + (j - center_y) ** 2 <= radius ** 2:
                    image[i, j] = 1.0
                    if add_patterns and np.random.rand() > 0.7:  
                        image[i, j] = 0.5 + 0.5 * np.sin(i + j)
    
    # Dairesel çizgi desenleri
    if add_patterns:
        for _ in range(3):  
            x1, y1 = np.random.randint(N, size=2)
            x2, y2 = np.random.randint(N, size=2)
            rr, cc = line(x1, y1, x2, y2)
            image[rr, cc] = 1.0 
    
    return image

def generate_detailed_rectangle_image(N, num_rectangles=5, add_patterns=True):
    image = np.random.rand(N, N) * 0.5  
    for _ in range(num_rectangles):
        start_x, start_y = np.random.randint(0, N // 2, size=2)
        width, height = np.random.randint(N // 10, N // 3, size=2)
        end_x = min(start_x + width, N)
        end_y = min(start_y + height, N)
        image[start_x:end_x, start_y:end_y] = 1.0
        if add_patterns:
            for _ in range(2):  
                rr, cc = np.random.randint(start_x, end_x), np.random.randint(start_y, end_y)
                if np.random.rand() > 0.5:
                    image[rr:rr + 2, cc:cc + 2] = 0.3 + 0.7 * np.random.rand()

    if add_patterns:
        for _ in range(3):
            x1, y1 = np.random.randint(N, size=2)
            x2, y2 = np.random.randint(N, size=2)
            rr, cc = line(x1, y1, x2, y2)
            image[rr, cc] = 0.5 + 0.5 * np.sin(np.arange(len(rr)))
    
    return image

base_path = "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/images_detailed"
os.makedirs(base_path, exist_ok=True)

N = 28  
num_images_per_class = 100

for i in range(num_images_per_class):
    image = generate_detailed_circle_image(N)
    filepath = os.path.join(base_path, f"class_A_image_{i+1:03}.txt")
    np.savetxt(filepath, image, fmt="%.2f")

for i in range(num_images_per_class):
    image = generate_detailed_rectangle_image(N)
    filepath = os.path.join(base_path, f"class_B_image_{i+1:03}.txt")
    np.savetxt(filepath, image, fmt="%.2f")

print(f"Detaylı görüntüler {base_path} klasörüne başarıyla kaydedildi.")
