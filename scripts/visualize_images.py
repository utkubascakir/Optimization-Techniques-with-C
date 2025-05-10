import os
import numpy as np
import matplotlib.pyplot as plt

base_path = "C:/Users/Utku/Desktop/Genel/Programlama/Optimization1/images_detailed" 

image_files = [f for f in os.listdir(base_path) if f.endswith(".txt")]

for image_file in image_files[120:125:1]:
    image_path = os.path.join(base_path, image_file)
    
    image = np.loadtxt(image_path)

    plt.imshow(image, cmap='gray', interpolation='nearest')
    plt.title(f"Görüntü: {image_file}")
    plt.colorbar()  
    plt.show()

print(f"Görüntüler {base_path} klasöründen başarıyla yüklendi ve görselleştirildi!")
