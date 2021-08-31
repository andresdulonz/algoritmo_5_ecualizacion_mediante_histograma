import cv2

img = cv2.imread('imagen_01.jpg')

print(img.shape)

# Modificar tamaño imagen por pixeles (Columna, fila)
# img_2 = cv2.resize(img, (300, 200))
# Modificar tamaño imagen por escala (Columna, fila)
img_2 = cv2.resize(img, None, fx=1.5, fy=0.5)

# Mostrar tamaño imagen (Fila, columna)
print(img_2.shape)

cv2.imshow('imagen', img_2)
cv2.waitKey()