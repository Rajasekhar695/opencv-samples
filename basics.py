import numpy as np
import cv2

'''Image Borders
image = cv2.imread('write.png')
replicate = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_REPLICATE)
cv2.imshow('Replicate Border', replicate)
reflect = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_REFLECT)
cv2.imshow('Replicate Border', reflect)
reflect101 = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_REFLECT_101)
cv2.imshow('Replicate Border', reflect101)
wrap = cv2.copyMakeBorder(image, 10, 10, 10, 10, cv2.BORDER_WRAP)
cv2.imshow('Replicate Border', wrap)'''

'''Splitting Image
image = cv2.imread('sample2.png')
#image[:, :, 2] = 0
#r = image[:, :, 2]
b, g, r = cv2.split(image)
image = cv2.merge((b, g, r))
cv2.imshow("image", image)'''


'''Image ROI
image = cv2.imread('sample2.png')
cv2.imshow("Original image", image)
fruit = image[402:540, 338:513]
image[100:238, 100:275] = fruit
cv2.imshow("Modified Image", image)'''

'''Accessing and modifying pixels
image = cv2.imread('sample2.png')
print(image.shape)
print(image.size)
print(image.dtype)
for i in range(100, 200):
    for j in range(100, 200):
        image[i, j] = [255, 255, 255]
cv2.imshow("Remade image", image)'''

'''Median Image
image = cv2.imread('sample2.png')
median = cv2.medianBlur(image, 7)
cv2.imshow("Median Image", median)'''

''' Averaging Image
image = cv2.imread('sample2.png')
kernel = np.ones((10, 10), np.float32)/100
dst = cv2.filter2D(image, -1, kernel)
cv2.imshow("Averaging Image", dst)'''

''' Edge Detecting
image = cv2.imread('write.png')
edge = cv2.Canny(image, 150, 250)
cv2.imshow("Original image", image)
cv2.imshow("Edge Detecting Image", edge)'''

''' GaissianBlur
src = cv2.imread('sample2.png')
dst = cv2.GaussianBlur(src, (7, 7), cv2.BORDER_DEFAULT)
cv2.imshow("Original image", src)
cv2.imshow("Gaussian Blur",dst)'''

'''BGR2GRAY
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Original image", image)
cv2.imshow("Gray image", gray)'''

'''Image Writing
cv2.imwrite('write.png', img)'''
cv2.waitKey(0)
cv2.destroyAllWindows()


