#!/usr/bin/env python
# coding=utf-8
import cv2
img = cv2.imread('target.jpg')
#print(img.shape)
result = cv2.resize(img, (1280, 1280))
cv2.imwrite('target_resize.png', result)
