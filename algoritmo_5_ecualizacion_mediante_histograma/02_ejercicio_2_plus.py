import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('imagen_02.jpg', 0)

# Modificar brillo multiplicando por escalar
# Compresion de pixeles grafica en decimales de entero:
# cercano a cero menos brillo
# lejano a cero mayor brillo
# n > 0
img_2 = cv2.multiply(img, 0.9)
img_3 = cv2.multiply(img, 0.8)
img_4 = cv2.multiply(img, 0.7)
img_5 = cv2.multiply(img, 0.6)

# Modificar brillo sumando un escalar
# Desplazamiento grafica: positivo izquierda mas brillo;
# negativo derecha menos brillo.
# -infinito < 0 < infinito
img_2 = cv2.add(img_2, -2)
img_3 = cv2.add(img_3, 0)
img_4 = cv2.add(img_4, 2)
img_5 = cv2.add(img_5, 4)

# Normalización del histograma por su tamaño
[M, N] = img.shape[0:2]

# Ecualización de la imagen
img_ecu = cv2.equalizeHist(img_2)

# Características del histograma: images, channels, mask, histSize, ranges, hist=..., accumulate=...)
# Arreglo de una dimención: .flatten()
hist = cv2.calcHist([img], [0], None, [256], [0,256]).flatten()/(M*N)
hist_2 = cv2.calcHist([img_ecu], [0], None, [256], [0,256]).flatten()/(M*N)
hist_3 = cv2.calcHist([img_3], [0], None, [256], [0,256]).flatten()/(M*N)
hist_4 = cv2.calcHist([img_4], [0], None, [256], [0,256]).flatten()/(M*N)
hist_5 = cv2.calcHist([img_5], [0], None, [256], [0,256]).flatten()/(M*N)

# Concatenacion de imagenes
img_res = np.hstack((img, img_ecu, img_3, img_4, img_5))

cv2.imshow('Imagenes', img_res)

# Mostrar el histograma
fig = plt.figure('Histograma multiple')
pl1 = fig.add_subplot(6,1,1)
plt.title('Imagen original')
plt.bar(range(len(hist)), hist)
pl2 = fig.add_subplot(6,1,2)
plt.title('Brillo multiplicado 0.9, sumado -2, ecualizado')
plt.bar(range(len(hist_2)), hist_2)
pl3 = fig.add_subplot(6,1,3)
plt.title('Brillo multiplicado 0.8, sumado 0')
plt.bar(range(len(hist_3)), hist_3)
pl2 = fig.add_subplot(6,1,4)
plt.title('Brillo multiplicado 0.7, sumado 2')
plt.bar(range(len(hist_4)), hist_4)
pl2 = fig.add_subplot(6,1,5)
plt.title('Brillo multiplicado 0.6, sumado 4')
plt.bar(range(len(hist_5)), hist_5)
pl2 = fig.add_subplot(6,1,6)
plt.title('Comparación cinco histogramas')
plt.bar(range(len(hist)), hist)
plt.bar(range(len(hist_2)), hist_2)
plt.bar(range(len(hist_3)), hist_3)
plt.bar(range(len(hist_4)), hist_4)
plt.bar(range(len(hist_5)), hist_5)

plt.show()

cv2.waitKey()