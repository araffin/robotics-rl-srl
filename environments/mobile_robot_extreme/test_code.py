
import os, glob, sys
import numpy as np
import cv2

img = cv2.imread("background.jpg")
print(img.shape)
cv2.imwrite("background_224.png", img)
img = cv2.resize(img, (128,128))
cv2.imwrite("background_128.png", img)
if __name__=="__main__":
    print("Start")
