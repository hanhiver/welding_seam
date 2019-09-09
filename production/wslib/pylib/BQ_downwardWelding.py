import math
import numpy as np 
import cv2 
import ctypes
import time
import sys

sys.path.append("..")
import wslib.clib.BQ_clib as clib 

BOTTOM_THICK = 80
NOISY_PIXELS = 20
BOTTOM_LINE_GAP_MAX = 5

"""
输入轮廓线图像得到焊缝最底部小平台中点位置。
bottom_thick: 底部小平台厚度
noisy_pixels: 作为噪音滤除的像素数目
cut_lines: 是否将底部小平台分成不同线段识别最长线段的中点位置。
"""
def getBottomCenter(lib, coreImage, bottom_thick = 30, noisy_pixels = 0, cut_lines = True):
    index = clib.coreLine2Index(lib, coreImage)
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
    index = clib.coreLine2Index(lib, coreImage)
    srt = index.argsort(kind = 'stable')
    idx = srt[:bottom_thick]

    bottom = index[idx]

    level = int(np.mean( bottom[noisy_pixels:(bottom.size - noisy_pixels)] ))
    center = int(np.mean( idx[noisy_pixels:(idx.size - noisy_pixels)] ))
    
    return center, level

"""
输入本帧画面得到的ceter值，经过平滑降噪计算之后输出。
目前的平滑算法是，当前值和前面三帧的平均值比较：
    超出合理范围： 丢弃不采用。
    超出预定范围： 平均化之后采用。
    未超出合理范围：直接采用。

调用此函数需要准备一个array存储此前多帧数据。

默认dropped_array=None表示不启用dropped value追踪功能。
如果需要启用该功能，需要提供一个array缓存此前丢弃的值。目前的逻辑会自动缓存三个丢弃的数据，
如果这三个数据都可以互相接近并且是被连续丢弃的，则改变当前的队列值，提供自动跟踪。
"""
def normalizeCenter(queue_array, center, queue_length = 10, thres_drop = 100, thres_normal = 50, move_limit = 3, skip = False, dropped_array = None):

    #print('normalizeCenter(queue_array = {}, center = {})'.format(queue_array, center))
    # 如果skip设置为真，不做处理，直接输出。
    if skip:
        return center, queue_array

    # 如果队列里没有填满数据，直接输出，不做处理。
    if len(queue_array) < queue_length:
        queue_array.append(center)
        return center, queue_array, dropped_array

    # 如果队列已经填满，可以开始处理数据。
    # 计算均值
    avg = 0; 
    for item in queue_array:
        avg += item

    avg = avg // len(queue_array)

    #array = np.array(queue_array)
    #avg = array.mean()
    delta = abs(center - avg)

    # 如果差值超过thres_drop，丢弃本次数据，返回之前的均值数据。
    if delta > thres_drop:
        print('Center {} DROPPED, avg: {}, array: {}'.format(center, avg, queue_array))
        dropped_array.append(center)

        if type(dropped_array) != type(None):
            # 如果连续Queue_length三分之一长度次丢弃数据，则认为跟踪丢失，
            # 将dropped_array的数据替换queue_array里的数据。
            if len(dropped_array) > (queue_length//3 - 1): 
                print("TRACK LOST, re-catch it!!!")
                queue_array = dropped_array.copy()
                dropped_array = []

        return avg, queue_array, dropped_array

    # 将本次数据添加到数据队列中，并将最早一次输入数据删除。
    #queue_array.append(center)
    #queue_array = queue_array[1:]
    
    # 如果差值超过thres_normal，输出和之前均值平均后结果。 
    if delta > thres_normal: 
        print('Center {} OVERED, avg: {}, array: {}'.format(center, avg, queue_array))
        queue_array.append(center)
        queue_array = queue_array[1:]
        # 如果本次有正常数据进入，则清空dropped_array
        if type(dropped_array) != type(None):
            dropped_array = []
        
        return (avg * 2 + center) // 3, queue_array, dropped_array
        #return (avg * 3 + center) // 4, queue_array

    # 将本次数据添加到数据队列中，并将最早一次输入数据删除。
    #queue_array.append(center)
    #queue_array = queue_array[1:]

    # 如果差值在可控范围内，直接输出。
    # 为了防止抖动，和之前一个信号比较，如果输出范围小于1，则不变化输出。
    # print("C: ", center, queue_array[-2])
    if abs(center - queue_array[-1]) < move_limit:
        center = queue_array[-1]

    # 将本次数据添加到数据队列中，并将最早一次输入数据删除。
    queue_array.append(center)
    queue_array = queue_array[1:]

    # 如果本次有正常数据进入，则清空dropped_array
    if type(dropped_array) != type(None):
        dropped_array = []

    #print('{}'.format(center))

    return center, queue_array, dropped_array

