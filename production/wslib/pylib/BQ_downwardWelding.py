import math
import numpy as np 
import cv2 
import ctypes
import time

"""
输入轮廓线图像得到焊缝最底部小平台中点位置。
bottom_thick: 底部小平台厚度
noisy_pixels: 作为噪音滤除的像素数目
cut_lines: 是否将底部小平台分成不同线段识别最长线段的中点位置。
"""
def getBottomCenter(lib, coreImage, bottom_thick = 30, noisy_pixels = 0, cut_lines = True):
    index = coreLine2Index(lib, coreImage)
    srt = index.argsort(kind = 'stable')
    
    idx = srt[:bottom_thick]
    idx.sort()

    if cut_lines:
        # 将所有在bottom范围内的线段分开。
        lines = []
        line = []
        for i in range(idx.size -1):
            if (abs(idx[i] - idx[i+1]) <= BOTTOM_LINE_GAP_MAX):
                line.append(idx[i])
            else:
                lines.append(line)
                line = []

        if len(line) != 0:
            lines.append(line)

        # 找到其中最长的线段。
        len_max = 0
        line_out = None
        for item in lines:
            if len(item) > len_max:
                len_max = len(item)
                line_out = item
        
        if len_max < bottom_thick//2:
            line_out = idx[noisy_pixels:bottom_thick-noisy_pixels]

    idx = np.array(line_out)
    bottom = index[idx]

    level = int(np.mean(bottom))
    center = int(np.mean(idx))
    bound = (idx.min(), idx.max())
    
    return center, level, bound

def getBottomCenter_old(lib, coreImage, bottom_thick = 30, noisy_pixels = 0):
    index = coreLine2Index(lib, coreImage)
    srt = index.argsort(kind = 'stable')
    idx = srt[:bottom_thick]

    bottom = index[idx]

    level = int(np.mean( bottom[noisy_pixels:(bottom.size - noisy_pixels)] ))
    center = int(np.mean( idx[noisy_pixels:(idx.size - noisy_pixels)] ))
    
    return center, level
