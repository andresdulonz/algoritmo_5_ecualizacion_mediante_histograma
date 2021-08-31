import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('imagen_01.jpg', 0)

# Normalización del histograma por su tamaño
[M, N] = img.shape[0:2]

# Características del histograma: images, channels, mask, histSize, ranges, hist=..., accumulate=...)
# Arreglo de una dimención: .flatten()
#hist = cv2.calcHist([img], [0], None, [256], [0,256]).flatten()/(M*N)
# Separación en quitiles
#hist = cv2.calcHist([img], [0], None, [5], [0,256]).flatten()/(M*N)

# Modificar brillo multiplicando por escalar
#img2= cv2.multiply(img, 0.5)

# Modificar brillo sumando un escalar
#img2= cv2.add(img, 50)

# Ecualización de la imagen
img_ecu = cv2.equalizeHist(img)

res = np.hstack((img, img_ecu))

hist = cv2.calcHist([res], [0], None, [256], [0,256]).flatten()/(M*N)

cv2.imshow('Imagen', res)

# Mostrar el histograma
fig = plt.figure()
plt.bar(range(len(hist)), hist)
plt.show()

cv2.waitKey()