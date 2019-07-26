import cv2 as cv
import numpy as np

a = [[0, 85, 127], [128, 180, 253], [0, 128, 255]]
a = np.array(a)

cv.imwrite('test.bmp', a)

