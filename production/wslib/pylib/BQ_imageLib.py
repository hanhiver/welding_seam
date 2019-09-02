import math
import numpy as np 
import cv2 
import ctypes
import time

"""
旋转图像，给定angle，旋转图像。
"""
def imgRotate(image, angle):
    # Get the dimention of the image and then determine the certer. 
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Get the rotation matrix. 
    M = cv2.getRotationMatrix2D(center = (cX, cY), angle = angle, scale = 1.0)

    # Perform the actrual rotation and return the image. 
    res = cv2.warpAffine(image, M, (w, h))

    return res

"""
获取图像中所有符合标准的线段。
min_length: 线段最短像素值。
max_line_gap: 线段之间最小跨越像素值
"""
def getLines(image, min_length = 100, max_line_gap = 25):
    kernel = np.ones((3,3),np.uint8)

    blur = cv2.medianBlur(image, 5)
    ret,binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations = 1)

    lines = cv2.HoughLinesP(closed, rho = 1, 
                           theta = np.pi/180, 
                           threshold = 100, 
                           minLineLength = min_length, 
                           maxLineGap = max_line_gap)

    return lines

"""
判断图像是否为正，如果不是，计算偏转角度以供后续函数矫正。
max_angle: 最大矫正角度
min_length:　最小表面线段长度。
max_line_gap:　最小线段之间跨越像素值。
"""
def getSurfaceAdjustAngle(image, max_angle = 10, min_length = 200, max_line_gap = 25):
    np.set_printoptions(precision=3, suppress=True)

    # Get the dimention of the image and then determine the certer. 
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    lines = getLines(image, min_length = min_length, max_line_gap = max_line_gap)

    zero_slope_lines_left = []
    zero_slope_lines_right = []
    max_radian = max_angle * np.pi / 180

    if type(lines) != type(None) and lines.size > 0:
        for line in lines: 

            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            theta = np.arctan((y2 - y1) / (x2 - x1))
            theta_apro = np.around(theta, 1)

            if theta_apro < max_radian and theta_apro > -max_radian:
            
                if (x1 + x2) < w:
                    zero_slope_lines_left.append([x1, y1, x2, y2, length, theta_apro, theta])
                else:
                    zero_slope_lines_right.append([x1, y1, x2, y2, length, theta_apro, theta])

    if zero_slope_lines_left or zero_slope_lines_right:

        reference_lines = []
        ret_radian_left = None
        ret_radian_right = None
        ret_radian = 0

        if zero_slope_lines_left:
            zero_slope_lines_left = np.array(zero_slope_lines_left)

            # Sort the lines with length. 
            index = np.argsort(zero_slope_lines_left, axis = 0)
            index_length = index[..., 4]
            zero_slope_lines_left = zero_slope_lines_left[index_length]

            # Get the longest X lines:
            x = zero_slope_lines_left.shape[0] // 4 + 1
            print(zero_slope_lines_left[::-1][:x])
            
            ret_radian_left = np.mean(zero_slope_lines_left[::-1][:x][..., 6])
            print('Radian LEFT: ', ret_radian_left)


        if zero_slope_lines_right:
            zero_slope_lines_right = np.array(zero_slope_lines_right)

            # Sort the lines with length. 
            index = np.argsort(zero_slope_lines_right, axis = 0)
            index_length = index[..., 4]
            zero_slope_lines_right = zero_slope_lines_right[index_length]

            # Get the longest X lines:
            x = zero_slope_lines_right.shape[0] // 4 + 1
            print(zero_slope_lines_right[::-1][:x])
            
            ret_radian_right = np.mean(zero_slope_lines_right[::-1][:x][..., 6])
            print('Radian RIGHT: ', ret_radian_right)

        if ret_radian_left and ret_radian_right:
            ret_radian = (ret_radian_right + ret_radian_left) / 2
        elif ret_radian_left:
            ret_radian = ret_radian_left
        elif ret_radian_right:
            ret_radian = ret_radian_right

        ret_angle = ret_radian * 180 / np.pi 

    else:
        print('Failed to found enough surface lines. ')
        ret_angle = 0

    return ret_angle 


"""
输入彩色图像，焊缝底部中点位置画出标志线。
"""
def drawTag(image, b_center, b_level, bottom_thick = None, bound = None):
    (h, w) = image.shape[:2]
    cv2.rectangle(image, (1, 1), (w-2, h-2), (130, 130, 130), 3)

    x1 = b_center
    x2 = b_center

    y1 = b_level - h//20 
    if y1 < 0:
        y1 = 0
    y2 = b_level + h//5 
    if y2 > h-1:
        y2 = h-1

    cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 3)
    if bound != None:
        cv2.line(image, (bound[0], y1 + 20), (bound[0], y2 - 20), (0, 255, 255), 1)
        cv2.line(image, (bound[1], y1 + 20), (bound[1], y2 - 20), (0, 255, 255), 1) 

    if bottom_thick != None:
        cv2.line(image, (x1 + BOTTOM_THICK//2, y1 + 30), (x2 + BOTTOM_THICK//2, y2 - 30), (0, 255, 0), 2)
        cv2.line(image, (x1 - BOTTOM_THICK//2, y1 + 30), (x2 - BOTTOM_THICK//2, y2 - 30), (0, 255, 0), 2)
