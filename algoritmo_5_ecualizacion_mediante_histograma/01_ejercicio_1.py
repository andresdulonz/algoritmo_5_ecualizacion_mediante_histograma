import cv2
import numpy as np
import matplotlib.pyplot as plt

img_o = cv2.imread('imagen_01.jpg')

# Modificar tamaño imagen por escala (Columna, fila)
img_med = cv2.resize(img_o, None, fx=0.50, fy=1)
img_crt = cv2.resize(img_o, None, fx=0.50, fy=0.50)

print(img_o.shape)
print(img_med.shape)
print(img_crt.shape)

# Normalización del histograma por su tamaño
[M, N] = img_o.shape[0:2]

# Características del histograma: images, channels, mask, histSize, ranges, hist=..., accumulate=...)
# Arreglo de una dimención: .flatten()
hist_o = cv2.calcHist([img_o], [0], None, [256], [0,256]).flatten()/(M*N)
hist_med = cv2.calcHist([img_med], [0], None, [256], [0,256]).flatten()/(M*N)
hist_crt = cv2.calcHist([img_crt], [0], None, [256], [0,256]).flatten()/(M*N)

# Concatenacion
#hist_f = np.hstack((hist_o, hist_med, hist_crt))

cv2.imshow('Imagen original', img_o)
cv2.imshow('Imagen media', img_med)
cv2.imshow('Imagen cuarto', img_crt)

# Mostrar el histograma
fig = plt.figure('Histograma comparativo')
plt.bar(range(len(hist_o)), hist_o)
plt.bar(range(len(hist_med)), hist_med)
plt.bar(range(len(hist_crt)), hist_crt)
plt.show()

cv2.waitKey()