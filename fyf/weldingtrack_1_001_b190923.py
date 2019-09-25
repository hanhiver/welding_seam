# 用法：用鼠标分别选取左右两侧两个ROI区域
# 每次选择完ROI区域，点击 q 键。
# ======================================
import cv2
import math
import numpy as np
from PIL import Image
from pylab import *
import matplotlib.pyplot as plt
import ctypes
import time
import argparse
#import arduino_serial as AS

global img
global point1, point2
global height_closed


    
# For Arduino Serial Communication. 
#import arduino_serial as AS

TEST_IMAGE = ('ssmall.png', 'sbig.png', 'rsmall.png')
#TEST_IMAGE = ('rsmall.png', )

WRITE_RESULT = False
RESIZE = 20
SLOPE_TH = 0.15

"""
图像对比度增强
Image_Gray: 输入灰度图像
Value: 图像对比度增大的值
"""

def Adjust_Image(Image_Gray, Value):
    Image_Adjust = Image_Gray * float(Value)
    #Image_Adjust = Image_Gray * Value
    Image_Adjust[Image_Adjust > 255] = 255
    Image_Adjust = np.round(Image_Adjust)
    Image_Adjust = Image_Adjust.astype(np.uint8)
    img = Image_Adjust
    return img

"""
使用移动平均法
"""
def Move_Mean(Index_Max_Chaque_Colomn):
    #records = np.zeros((10, 1))
    width = len(Index_Max_Chaque_Colomn)
    A = Index_Max_Chaque_Colomn
    
    # 使用移动平均法
    for i in range(20, width - 21):
        records = Index_Max_Chaque_Colomn[i - 20:i + 20]
        mean = get_average(records[0:39])
        deviation = get_standard_deviation(records[0:39])
        if abs(records[39]-records[38]) > 4*deviation:
            A[i] = np.round(mean)
        else:
            mean = get_average(records)
            A[i] = np.round(mean)
    return A

"""
窗口过滤
A : 经过平滑后的激光曲线在图像中的位置值。
"""
def Window_Filter(A, Window_Half_Height, cut_img):
    # 利用平滑以后的曲线再次选取最大值。类似一个窗口功能
    # 这回选取的最大值是在曲线附近一个邻域内选取。
    (height, width) = cut_img.shape[:2]
    for i in range(width):
        A[i] = [A[i] - Window_Half_Height + np.argmax(cut_img[int(A[i]-Window_Half_Height):int(A[i]+ Window_Half_Height), i])]
    return A


"""
平均值
"""
def get_average(records):
#    mean = np.round(sum(records) / len(records))
    mean = sum(records) / len(records)
    #return mean[0]
    return mean
"""
方差 反映一个数据集的离散程度
"""
def get_variance(records):
    average = get_average(records)
    return sum([(x - average)** 2 for x in records]) / len(records)
    
"""
标准差 == 均方差 反映一个数据集的离散程度
"""
def get_standard_deviation(records):
    variance = get_variance(records)
    return math.sqrt(variance)



"""
* Savitzky-Golay平滑滤波函数
* data: list格式的1*n 维数据
* window_size: 拟合的窗口大小
* rank: 拟合多项式阶次
* ndata: 修正后的值
"""
def savgol(data, window_size, rank):
    m = int((window_size - 1)/2)
    odata = data[:]
    # 处理边缘数据，首尾增加 m 个首尾项。
    for i in range(m):
        odata.insert(0, odata[0])
        odata.insert(len(odata), odata[len(odata)-1])
    #创建X矩阵
    x = create_x(m, rank)
    #计算加权系数矩阵B
    b = (x * (x.T * x).I) * x.T
    a0 = b[m]
    a0 = a0.T
    #计算平滑修正后的值
    ndata = []
    for i in range(len(data)):
        y = [odata[i + j] for j in range(window_size)]
        y1 = (np.mat(y)).T * a0
        y1 = float(y1)
        ndata.append(y1)
    return ndata

"""
Savitzky-Golay平滑滤波去噪
* 创建系数矩阵X
* size: 2*size +1 = window_size
* rank: 拟合多项式阶次
* x : 创建的系数矩阵 
"""
def create_x(size, rank):
    x = []
    for i in range(2*size + 1):
        m = i - size
        row = [m**j for j in range(rank)]
        x.append(row)
    x = np.mat(x)
    return x


"""
鼠标事件(选择ROI)
"""
def on_mouse(event, x, y, flags, param):
    global img, point1, point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        point1 = (x, y)
        cv2.circle(img2, point1, 5, (0, 255, 0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        cv2.rectangle(img2, point1, (x, y), (255, 0, 0), 5)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:
        point2 = (x, y)
        cv2.rectangle(img2, point1, point2, (255, 0, 0), 5)
        cv2.imshow('image', img2)

"""
全局阈值
"""
def threshold_demo(image):
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # 把输入图像灰度化
    # 直接阈值化是对输入的单通道矩阵逐像素进行阈值分割
    ret, binary = cv2.threshold(image, 0, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
    #print("threshold value %s" % ret)
    #cv2.namedWindow("binary0", cv2.WINDOW_NORMAL)
    #cv2.imshow("binary0", binary)
    return binary

"""
提取线结构光中心
cut_img: ROI范围内的图像。
"""
def Abstract_Center_Line(cut_img):
    
    #获取图片的宽度和高度
    width, height = cut_img.shape[:2][::-1]
    # 提取线结构光中心
    Index_Max_Chaque_Colomn = np.zeros((width, 1))

    #print("Cut_Image Shape:{}".format(np.shape(cut_img)))
    #找出每列中最大值所在的位置（图2）。这个位置认为是线激光的位置。但是由于噪声的影响，有一些偏差。
    #plt.imshow(cut_img)
    for i in range(width):
        Num = 0
        Sum = 0
        # 该列中所有点中最大的灰度值。
        Max_Colomn = max(cut_img[:, i])

        for j in range(height):

            if cut_img[j, i] > (Max_Colomn - 20):
                # 统计出所有灰度值在（Max_Colomn-20, Max_Colomn）之间的点的数量。
                Num = Num + 1
                # 计算出这些点的纵坐标的值的总和
                Sum = Sum + j
        # 求出其平均纵坐标的位置值。
#        Index_Max_Chaque_Colomn[i] = np.round(Sum/Num)
        Index_Max_Chaque_Colomn[i] = Sum/Num
        #Index_Max_Chaque_Colomn[i] = np.argmax(cut_img[:, i])
        #将所有Index_Max_Chaque_Colomn中为零的值，替换掉
        if Index_Max_Chaque_Colomn[i] == 0:
            Index_Max_Chaque_Colomn[i] = Index_Max_Chaque_Colomn[i-1]
        #plt.plot(i, Index_Max_Chaque_Colomn[i], '*', 'Color', 'blue') 
    #plt.show()
    return Index_Max_Chaque_Colomn

"""
通过修改 getSurfaceAdjustAngle 函数，求出两侧焊缝拐点的位置
max_angle: 最大矫正角度
min_length:　最小表面线段长度。
max_line_gap:　最小线段之间跨越像素值。
"""
def FindWeldingKeyPoint(cut_img):

     #＝＝＝＝＝＝＝＝＝ 首先对ＲＯＩ区域，进行霍夫变化，提取ＲＯＩ中的水平方向的直线＝＝＝＝＝＝＝＝＝＝＝
    blur = cv2.medianBlur(cut_img, 5)

    kernel = np.ones((3,3),np.uint8)
    ret, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    #cv2.namedWindow('closed', cv2.WINDOW_NORMAL)
    #cv2.imshow('closed', closed)
    #cv2.waitKey(0)
    
    # 类似于韩老师的算法中的 getLines 函数　－－－－－－－－
    (height_closed, width_closed) = closed.shape[:2]
    min_length = width_closed // 20
    max_line_gap = 2
    lines = cv2.HoughLinesP(closed, rho = 1, 
                           theta = np.pi/180, 
                           threshold = 100, 
                           minLineLength = min_length, 
                           maxLineGap=max_line_gap)
    
    zero_slope_lines_left = []
    zero_slope_lines_right = []
    zero_slope_lines_middle_left = []
    zero_slope_lines_middle_right = []
    max_angle = 30
    max_radian = max_angle * np.pi / 180
    line_longest_left = np.zeros((1, 4))
    line_longest_right = np.zeros((1, 4))
    line_longest_middle_left = np.zeros((1, 4))
    line_longest_middle_right = np.zeros((1, 4))
    x_1_left = 0
    y_1_left = 0
    x_1_right = 0
    y_1_right = 0
    x_2_left = 0
    y_2_left = 0
    x_2_right = 0
    y_2_right = 0

    if type(lines) != type(None) and lines.size > 0:
        for line in lines: 

            # 计算出lines中所有线段的长度，倾斜角度
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x1 - x2)** 2 + (y1 - y2)** 2)
            if (x2 != x1):
                theta = np.arctan((y2 - y1) / (x2 - x1))
            else:
                theta = 0
            theta_apro = np.around(theta, 1)

            if theta_apro < max_radian and theta_apro > -max_radian:
            
                # 如果线段两端的端点X坐标值，其和小于width_closed，说明这条直线是在图像的左侧。
                # 说明这个鼠标选取的是左侧的ＲＯＩ
                if (x1 + x2) < width_closed:
                    # 左侧的线段放在 zero_slope_lines_left 中。
                    zero_slope_lines_left.append([x1, y1, x2, y2, length, theta_apro, theta])
                else:
                    # 右侧的线段放在zero_slope_lines_right中。
                    zero_slope_lines_right.append([x1, y1, x2, y2, length, theta_apro, theta])

    if zero_slope_lines_left or zero_slope_lines_right:

        #reference_lines = []
        ret_radian_left = None
        ret_radian_right = None
        ret_radian = 0

        # 对于在鼠标左侧提取出来的水平直线段：
        if zero_slope_lines_left:
            zero_slope_lines_left = np.array(zero_slope_lines_left)

            # Sort the lines with length. 按照长度对线段进行排序
            index = np.argsort(zero_slope_lines_left, axis = 0)
            index_length = index[..., 4]
            zero_slope_lines_left = zero_slope_lines_left[index_length]

            # Get the longest X lines:
            x = zero_slope_lines_left.shape[0] // 4 + 1
            #print(zero_slope_lines_left[::-1][:x])

            ret_radian_left = np.mean(zero_slope_lines_left[::-1][:x][..., 6])
            #print('Radian LEFT: ', ret_radian_left)

            # 显示出左侧最长的线段-------------------------------------------
            # # 找出的最长的直线
            line_longest_left = zero_slope_lines_left[::-1][:x]
            
            x_1_left = line_longest_left[0][0]
            y_1_left = line_longest_left[0][1]
            x_2_left = line_longest_left[0][2]
            y_2_left = line_longest_left[0][3]
            
            x_left = [line_longest_left[0][0], line_longest_left[0][2]]
            y_left = [line_longest_left[0][1], line_longest_left[0][3]]

            #plt.imshow(closed)
            
            #for i in range(len(x_left)):
            #    plt.plot(x_left[i], y_left[i], color='r')
            #    plt.scatter(x_left[i], y_left[i], color='b')
            #plt.show()
            
        if zero_slope_lines_right:
            zero_slope_lines_right = np.array(zero_slope_lines_right)

            # Sort the lines with length. 
            index = np.argsort(zero_slope_lines_right, axis = 0)
            index_length = index[..., 4]
            zero_slope_lines_right = zero_slope_lines_right[index_length]

            # Get the longest X lines:
            x = zero_slope_lines_right.shape[0] // 4 + 1
            #print(zero_slope_lines_right[::-1][:x])
            
            ret_radian_right = np.mean(zero_slope_lines_right[::-1][:x][..., 6])
            #print('Radian RIGHT: ', ret_radian_right)

            # 显示出右侧最长的线段-------------------------------------------
            # 找出的最长的直线
            line_longest_right = zero_slope_lines_right[::-1][:x]
            x_1_right = line_longest_right[0][0]
            y_1_right = line_longest_right[0][1]
            x_2_right = line_longest_right[0][2]
            y_2_right = line_longest_right[0][3]
            
            x_right = [line_longest_right[0][0], line_longest_right[0][2]]
            y_right = [line_longest_right[0][1], line_longest_right[0][3]]
                   
    #print('height_closed', height_closed)

    # 当左右两侧都能找到直线的时候;
    if len(zero_slope_lines_left)*len(zero_slope_lines_right):
        
        # 利用所求得的两条直线的相近的两个端点，将图像的中间部分，焊缝坡口部分切出来
        # 焊缝坡口靠左，四分之一坡口宽度的图像为：

        #＝＝＝＝ 修改错误＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        # 错误的原因是：有时找到的右侧的直线的第一个端点的坐标值　x_1_right 　小于　 x_2_left，
        # 导致无法取得图像。
        if (type(zero_slope_lines_left) != type(None)) and (len(zero_slope_lines_left) > 0) and ((x_1_right - x_2_left) > 20):

            cut_img_middle_left = blur[0:height_closed, int(x_2_left):int((x_1_right + 3 * x_2_left) / 4)]

            kernel = np.ones((3, 3), np.uint8)
            ret, cut_img_middle_left_binary = cv2.threshold(cut_img_middle_left, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            cut_img_middle_left_closed = cv2.morphologyEx(cut_img_middle_left_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # 焊缝坡口靠左，从图像中找到最长的直线
            (height_cut_img_middle_left_closed, width_cut_img_middle_left_closed) = cut_img_middle_left_closed.shape[:2]
            min_length = width_cut_img_middle_left_closed // 40
            max_line_gap = 20
            lines_middle_left = cv2.HoughLinesP(cut_img_middle_left_closed, rho = 1, 
                           theta = np.pi/180, 
                           threshold = 5, 
                           minLineLength = min_length, 
                           maxLineGap=max_line_gap)
            zero_slope_lines_middle_left = []
            max_angle = 75
            max_radian = max_angle * np.pi / 180
            
            if type(lines_middle_left) != type(None) and lines_middle_left.size > 0:
                
                for line in lines_middle_left:
                    
                    # 计算出lineslines_middle_left 中所有线段的长度，倾斜角度
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x1 - x2)** 2 + (y1 - y2)** 2)
                    if (x2 != x1):
                        theta = np.arctan((y2 - y1) / (x2 - x1))
                    else:
                        theta = 0
                    theta_apro = np.around(theta, 1)
                    
                    if theta_apro < max_radian and theta_apro > -max_radian:
                        zero_slope_lines_middle_left.append([x1, y1, x2, y2, length, theta_apro, theta])
                        
                # 如果　zero_slope_lines_middle_left　不为空的话
                if zero_slope_lines_middle_left:
                    #reference_lines = []
                    ret_radian_left = None
                    ret_radian_right = None
                    ret_radian = 0
                    
                    zero_slope_lines_middle_left = np.array(zero_slope_lines_middle_left)
                    # Sort the lines with length. 按照长度对线段进行排序
                    index = np.argsort(zero_slope_lines_middle_left, axis=0)
                    index_length = index[..., 4]
                    zero_slope_lines_middle_left = zero_slope_lines_middle_left[index_length]
                    # Get the longest X lines:
                    x = zero_slope_lines_middle_left.shape[0] // 4 + 1
                    #print(zero_slope_lines_middle_left[::-1][:x])
                    
                    ret_radian_left = np.mean(zero_slope_lines_middle_left[::-1][:x][..., 6])
                    #print('Radian LEFT: ', ret_radian_left)
                    
                    # 显示出左侧最长的线段-------------------------------------------
                    # # 找出的最长的直线
                    line_longest_middle_left = zero_slope_lines_middle_left[::-1][:x]
                    x_1_middle_left = line_longest_middle_left[0][0]
                    y_1_middle_left = line_longest_middle_left[0][1]
                    x_2_middle_left = line_longest_middle_left[0][2]
                    y_2_middle_left = line_longest_middle_left[0][3]
                    
                    x_middle_left = [line_longest_middle_left[0][0], line_longest_middle_left[0][2]]
                    y_middle_left = [line_longest_middle_left[0][1], line_longest_middle_left[0][3]]
            

        # 焊缝坡口靠右侧，四分之一坡口宽度的图像为：
        if type(zero_slope_lines_right) != type(None) and len(zero_slope_lines_right) > 0 and ((x_1_right - x_2_left) > 20):
            # 焊缝坡口靠左，四分之一坡口宽度的图像为：
            cut_img_middle_right = blur[0:height_closed, int((x_2_left + 3*x_1_right)/4):int(x_1_right )]
            
            kernel = np.ones((3, 3), np.uint8)
            ret, cut_img_middle_right_binary = cv2.threshold(cut_img_middle_right, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            cut_img_middle_right_closed = cv2.morphologyEx(cut_img_middle_right_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # 焊缝坡口靠左，从图像中找到最长的直线
            (height_cut_img_middle_right_closed, width_cut_img_middle_right_closed) = cut_img_middle_right_closed.shape[:2]
            min_length = width_cut_img_middle_right_closed // 40
            max_line_gap = 20
            lines_middle_right = cv2.HoughLinesP(cut_img_middle_right_closed, rho = 1, 
                           theta = np.pi/180, 
                           threshold = 5, 
                           minLineLength = min_length, 
                           maxLineGap=max_line_gap)
            zero_slope_lines_middle_right = []
            max_angle = 75
            max_radian = max_angle * np.pi / 180
            
            if type(lines_middle_right) != type(None) and lines_middle_right.size > 0:
                
                for line in lines_middle_right:
                    
                    # 计算出lineslines_middle_right 中所有线段的长度，倾斜角度
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x1 - x2)** 2 + (y1 - y2)** 2)
                    if (x2 != x1):
                        theta = np.arctan((y2 - y1) / (x2 - x1))
                    else:
                        theta = 0
                    theta_apro = np.around(theta, 1)
                    
                    if theta_apro < max_radian and theta_apro > -max_radian:
                        zero_slope_lines_middle_right.append([x1, y1, x2, y2, length, theta_apro, theta])
                        
                # 如果　zero_slope_lines_middle_right　不为空的话
                if zero_slope_lines_middle_right:
                    #reference_lines = []
                    #ret_radian_left = None
                    ret_radian_right = None
                    ret_radian = 0
                    
                    zero_slope_lines_middle_right = np.array(zero_slope_lines_middle_right)
                    # Sort the lines with length. 按照长度对线段进行排序
                    index = np.argsort(zero_slope_lines_middle_right, axis=0)
                    index_length = index[..., 4]
                    zero_slope_lines_middle_right = zero_slope_lines_middle_right[index_length]
                    # Get the longest X lines:
                    x = zero_slope_lines_middle_right.shape[0] // 4 + 1
                    #print(zero_slope_lines_middle_right[::-1][:x])
                    
                    ret_radian_right = np.mean(zero_slope_lines_middle_right[::-1][:x][..., 6])
                    #print('Radian LEFT: ', ret_radian_right)
                    
                    # 显示出右侧最长的线段-------------------------------------------
                    # # 找出的最长的直线
                    line_longest_middle_right = zero_slope_lines_middle_right[::-1][:x]
                    x_1_middle_right = line_longest_middle_right[0][0]
                    y_1_middle_right = line_longest_middle_right[0][1]
                    x_2_middle_right = line_longest_middle_right[0][2]
                    y_2_middle_right = line_longest_middle_right[0][3]
                    
                    x_middle_right = [line_longest_middle_right[0][0], line_longest_middle_right[0][2]]
                    y_middle_right = [line_longest_middle_right[0][1], line_longest_middle_right[0][3]]
            
                    
    
    # 得出的左右两侧焊缝拐点附近的４条直线的端点
    if len(zero_slope_lines_middle_left) != 0 and len(zero_slope_lines_middle_right) != 0:
        x = [line_longest_left[0][0], line_longest_left[0][2],
                    line_longest_right[0][0], line_longest_right[0][2],
                    line_longest_middle_left[0][0] + x_2_left, line_longest_middle_left[0][2] +x_2_left,
                    line_longest_middle_right[0][0] + int((x_2_left + 3*x_1_right)/4), line_longest_middle_right[0][2] + int((x_2_left + 3*x_1_right)/4)]
        y = [line_longest_left[0][1], line_longest_left[0][3],
                     line_longest_right[0][1], line_longest_right[0][3],
                     line_longest_middle_left[0][1], line_longest_middle_left[0][3],
                     line_longest_middle_right[0][1], line_longest_middle_right[0][3]]
    elif len(zero_slope_lines_middle_left) == 0 or len(zero_slope_lines_middle_right) == 0:
        x = [line_longest_left[0][0], line_longest_left[0][2],
                    line_longest_right[0][0], line_longest_right[0][2],
                    x_2_left, x_2_left,
                    x_1_right, x_1_right]
        y = [line_longest_left[0][1], line_longest_left[0][3],
                     line_longest_right[0][1], line_longest_right[0][3],
                     y_2_left, y_2_left, y_1_right, y_1_right]


    
    

    #print('x0 - x7 ', x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7])
    #print('y0 - y7 ', y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7])

    
    # 分别求出两条直线的交点
    if (x[5] != x[4]) and (x[1] != x[0]) and (y[5] != y[4]) and (y[1] != y[0]) and (x[7] != x[6]) and (x[3] != x[2]) and (y[7] != y[6]) and (y[3] != y[2]) and (((y[3] - y[2]) / (x[3] - x[2]) - (y[7] - y[6]) / (x[7] - x[6])) != 0) and (((x[3] - x[2]) / (y[3] - y[2]) - (x[7] - x[6]) / (y[7] - y[6])) != 0) :
        x_Key_left = (y[4] - y[0] - x[4] * (y[5] - y[4]) / (x[5] - x[4]) + x[0] * (y[1] - y[0]) / (x[1] - x[0])) / ((y[1] - y[0]) / (x[1] - x[0]) - (y[5] - y[4]) / (x[5] - x[4]))
        y_Key_left = (x[4] - x[0] - y[4] * (x[5] - x[4]) / (y[5] - y[4]) + y[0] * (x[1] - x[0]) / (y[1] - y[0])) / ((x[1] - x[0]) / (y[1] - y[0]) - (x[5] - x[4]) / (y[5] - y[4]))
        x_Key_right = (y[6] - y[2] - x[6] * (y[7] - y[6]) / (x[7] - x[6]) + x[2] * (y[3] - y[2]) / (x[3] - x[2])) / ((y[3] - y[2]) / (x[3] - x[2]) - (y[7] - y[6]) / (x[7] - x[6]))
        y_Key_right = (x[6] - x[2] - y[6] * (x[7] - x[6]) / (y[7] - y[6]) + y[2] * (x[3] - x[2]) / (y[3] - y[2])) / ((x[3] - x[2]) / (y[3] - y[2]) - (x[7] - x[6]) / (y[7] - y[6]))

        x_Points_Key = [x_Key_left, x_Key_right]
        y_Points_Key = [y_Key_left, y_Key_right]
    
    # 当左右两侧的直线的端点的纵坐标相等时：
    else:
        if ((y[2]==y[3]) and (y[0] != y[1])) and (x[5] != x[4]) and (y[5] != y[4]) and (x[1] != x[0]) and (y[7] != y[6]):
            x_Key_left = (y[4] - y[0] - x[4] * (y[5] - y[4]) / (x[5] - x[4]) + x[0] * (y[1] - y[0]) / (x[1] - x[0])) / ((y[1] - y[0]) / (x[1] - x[0]) - (y[5] - y[4]) / (x[5] - x[4]))
            y_Key_left = (x[4] - x[0] - y[4] * (x[5] - x[4]) / (y[5] - y[4]) + y[0] * (x[1] - x[0]) / (y[1] - y[0])) / ((x[1] - x[0]) / (y[1] - y[0]) - (x[5] - x[4]) / (y[5] - y[4]))
            x_Key_right = x[6]+(y[3]-y[6])*(x[7]-x[6])/(y[7]-y[6])
            y_Key_right = y[2]
        elif ((y[2]!=y[3]) and (y[0] == y[1])) and (x[5] != x[4]) and (y[5] != y[4]) and (x[7] != x[6]) and (x[3] != x[2]) and (y[7] != y[6]):
            x_Key_left = x[4]+(y[1]-y[4])*(x[5]-x[4])/(y[5]-y[4])
            y_Key_left = y[0]
            x_Key_right = (y[6] - y[2] - x[6] * (y[7] - y[6]) / (x[7] - x[6]) + x[2] * (y[3] - y[2]) / (x[3] - x[2])) / ((y[3] - y[2]) / (x[3] - x[2]) - (y[7] - y[6]) / (x[7] - x[6]))
            y_Key_right = (x[6] - x[2] - y[6] * (x[7] - x[6]) / (y[7] - y[6]) + y[2] * (x[3] - x[2]) / (y[3] - y[2])) / ((x[3] - x[2]) / (y[3] - y[2]) - (x[7] - x[6]) / (y[7] - y[6]))
        elif ((y[2] == y[3]) and (y[0] == y[1])) and (y[5] != y[4]) and (y[7] != y[6]):
            x_Key_left = x[4]+(y[1]-y[4])*(x[5]-x[4])/(y[5]-y[4])
            y_Key_left = y[0]
            x_Key_right = x[6]+(y[3]-y[6])*(x[7]-x[6])/(y[7]-y[6])
            y_Key_right = y[2]
        else:
            x_Key_left = x[1]
            y_Key_left = y[1]
            x_Key_right = x[2]
            y_Key_right = y[2]

        x_Points_Key = [x_Key_left, x_Key_right]
        y_Points_Key = [y_Key_left, y_Key_right]

    return x_Key_left, y_Key_left, x_Key_right, y_Key_right


def bin2Hex(num):
    hex_num = hex(num)
    if(len(hex_num) == 3):
        hex_num = '000'+hex_num[2:]
    elif(len(hex_num) == 4):
        hex_num = '00'+hex_num[2:]
    elif(len(hex_num) == 5):
        hex_num = '0'+hex_num[2:]
    elif(len(hex_num) == 6):
        hex_num = hex_num[2:]
    return hex_num

"""
十进制转4字节十六进制，前两位为高位后两位为低位
num：输入十进制数字
"""
def bin2Hex(num):
    hex_num = hex(num)
    if(len(hex_num) == 3):
        hex_num = '000'+hex_num[2:]
    elif(len(hex_num) == 4):
        hex_num = '00'+hex_num[2:]
    elif(len(hex_num) == 5):
        hex_num = '0'+hex_num[2:]
    elif(len(hex_num) == 6):
        hex_num = hex_num[2:]
    return hex_num



"""
绘制十字
img 目标图像
point 十字中心点
color 颜色 (255, 255, 255)
size 十字尺寸
thickness 粗细
"""
def DrawCross(img, point_x, point_y, size, color, thickness):
    cv2.line(img, (int(point_x), int(point_y) - size), (int(point_x), int(point_y) + size), color, thickness)
    cv2.line(img, (int(point_x) - size, int(point_y)), (int(point_x) + size, int(point_y)), color, thickness)

"""
输入本帧画面得到的center值，经过平滑降噪计算之后输出。
目前的平滑算法是，当前值和前面三帧的平均值比较：
超出合理范围： 丢弃不采用。
超出预定范围： 平均化之后采用。
未超出合理范围：直接采用。
    
调用此函数需要准备一个array存储此前多帧数据。
输入参数：
center_array:  前面定义的空数组
b_center: 底部的中心点位置。
skip=False
"""
def normalizeCenter(queue_array_x, queue_array_y, point_x, point_y, queue_length=10, thres_drop=60, thres_normal=25):
    #queue_array_x = queue_array_x.tolist()
    #queue_array_y = queue_array_y.tolist()

    # queue_array 队列数组
    # # 
    # # 如果队列里没有填满数据，直接输出，不做处理。
    if (len(queue_array_x) < queue_length) and (len(queue_array_y) < queue_length):
        queue_array_x.append(point_x)
        queue_array_y.append(point_y)
        #queue_array_x = np.array(queue_array_x)
        #queue_array_y = np.array(queue_array_y)
        return point_x, point_y, queue_array_x, queue_array_y
        
    # 如果队列已经填满，可以开始处理数据。
    # 计算均值
    avg_x = 0; 
    for item in queue_array_x:
        avg_x += item
    
    avg_y = 0; 
    for item in queue_array_y:
        avg_y += item
    
    avg_x = avg_x // len(queue_array_x)
    avg_y = avg_y // len(queue_array_y)
    
    #array = np.array(queue_array)
    #avg = array.mean()
    #delta = abs(point - avg)
    delta = ((point_x - avg_x)**2 + (point_y - avg_y)**2)**0.5
    
    # 如果差值超过thres_drop，丢弃本次数据，返回之前的均值数据。
    if delta > thres_drop:
        #print('Center {} DROPED, avg: {}, array: {}'.format(center, avg, queue_array))
        #queue_array_x = np.array(queue_array_x)
        #queue_array_y = np.array(queue_array_y)
        return avg_x, avg_y, queue_array_x, queue_array_y
    
    # 将本次数据添加到数据队列中，并将最早一次输入数据删除。
    queue_array_x.append(point_x)
    queue_array_y.append(point_y)
    queue_array_x = queue_array_x[1:]
    queue_array_y = queue_array_y[1:]
        
    # 如果差值超过thres_normal，输出和之前均值平均后结果。    
    if delta > thres_normal:
        #queue_array_x = np.array(queue_array_x)
        #queue_array_y = np.array(queue_array_y)
        #print('Center {} OVERED, avg: {}, array: {}'.format(center, avg, queue_array))
         #return (avg_x  + point_x*queue_length) // (queue_length+1), (avg_y + point_y*queue_length) // (queue_length+1), queue_array_x, queue_array_y
        return (avg_x*2  + point_x) // 3, (avg_y*2 + point_y) // 3, queue_array_x, queue_array_y
    
    # 如果差值在可控范围内，直接输出。
    #queue_array_x = np.array(queue_array_x)
    #queue_array_y = np.array(queue_array_y)
    return point_x, point_y, queue_array_x, queue_array_y


"""
韩老师的函数＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
从原始输入图像得到最终识别结果的轮廓线。
black_limit: 小于等于这个值的点会被视作黑色。
correct_angle: 是否对图像做矫正处理。 
"""
def getLineImage(lib, image, black_limit = 0, correct_angle = True):
    (h, w) = image.shape[:2]
    
    if correct_angle:
        kernel = np.ones((5,5),np.uint8)

        angle = getSurfaceAdjustAngle(image, min_length = 200//RESIZE)

        #print('Rotate angle: ', angle)

        #print('Before rotation: ', image.shape)
        image = imgRotate(image, angle)
        #print('After rotation: ', image.shape)

        #level = getSurfaceLevel(image, min_length = 200//RESIZE)[:2]
        # ＝＝＝＝＝＝＝＝＝＝用下面两个函数替换上面的getSurfaceLevel函数，用于加速实验＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        lines, image, max_angle, min_length, max_line_gap = getSurfaceLevel_Part_One(image, min_length=200 // RESIZE)
        level = getSurfaceLevel_Part_Two(lines, image, max_angle, min_length, max_line_gap)[:2]
        
        #print('Surface Level: ', level)

    level = (h//2, h//2)

#    start = time.clock()
    coreImage = getCoreImage(lib, image, black_limit = black_limit)
    lineImage = followCoreLine(lib, coreImage, level, min_gap = 100//RESIZE, black_limit = black_limit)
#    end = time.clock()
    #print("TIME COST: ", end - start, ' seconds')

    return lineImage

"""
韩老师的函数＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
对于一个图像，得到由核心点组成的核心图。
black_limit: 小于等于这个值的点会被视作黑色。
"""
def getCoreImage(lib, image, black_limit = 0):
    (h, w) = image.shape[:2]

    src = np.ctypeslib.as_ctypes(image)
    dst = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_uint8) * w * h)
    
    lib.getCoreImage(src, dst, h, w, black_limit)
    
    dst = ctypes.cast(dst, ctypes.POINTER(ctypes.c_uint8))
    coreImage = np.ctypeslib.as_array(dst, shape = image.shape)

    return coreImage


"""
韩老师的函数＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
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
            length = np.sqrt((x1 - x2)** 2 + (y1 - y2)** 2)
            if (x2 != x1):
                theta = np.arctan((y2 - y1) / (x2 - x1))
            else:
                theta = 0
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
            #print(zero_slope_lines_left[::-1][:x])
            
            ret_radian_left = np.mean(zero_slope_lines_left[::-1][:x][..., 6])
            #print('Radian LEFT: ', ret_radian_left)


        if zero_slope_lines_right:
            zero_slope_lines_right = np.array(zero_slope_lines_right)

            # Sort the lines with length. 
            index = np.argsort(zero_slope_lines_right, axis = 0)
            index_length = index[..., 4]
            zero_slope_lines_right = zero_slope_lines_right[index_length]

            # Get the longest X lines:
            x = zero_slope_lines_right.shape[0] // 4 + 1
            #print(zero_slope_lines_right[::-1][:x])
            
            ret_radian_right = np.mean(zero_slope_lines_right[::-1][:x][..., 6])
            #print('Radian RIGHT: ', ret_radian_right)

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
韩老师的函数＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
得到焊接表面到底面深度
"""
def getSurfaceLevel(image, max_angle = 5, min_length = 200, max_line_gap = 25):
    np.set_printoptions(precision=3, suppress=True)

    lines = getLines(image, min_length=min_length, max_line_gap=max_line_gap)
    
    # Get the dimention of the image and then determine the certer. 
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    ret_level_left = cX
    ret_level_right = cX
    ret_bevel_top_left = -1
    ret_bevel_top_right = -1

    zero_slope_lines_left = []
    zero_slope_lines_right = []
    max_radian = max_angle * np.pi / 180

    #print('No LINE: ', type(lines))

    if type(lines) != type(None) and lines.size > 0:
        for line in lines: 

            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x1 - x2)** 2 + (y1 - y2)** 2)
            if (x2 != x1):
                theta = np.arctan((y2 - y1) / (x2 - x1))
            else:
                theta = 0
            theta_apro = np.around(theta, 1)

            if theta_apro < max_radian and theta_apro > -max_radian:
            
                if (x1 + x2) < w:
                    zero_slope_lines_left.append([x1, y1, x2, y2, length, theta_apro, theta])
                else:
                    zero_slope_lines_right.append([x1, y1, x2, y2, length, theta_apro, theta])

    if zero_slope_lines_left or zero_slope_lines_right:

        reference_lines = []
        ret_level_left = -1
        ret_level_right = -1

        if zero_slope_lines_left:
            zero_slope_lines_left = np.array(zero_slope_lines_left)

            # Sort the lines with length. 
            index = np.argsort(zero_slope_lines_left, axis = 0)
            index_length = index[..., 4]
            zero_slope_lines_left = zero_slope_lines_left[index_length]

            # Get the longest X lines:
            x = zero_slope_lines_left.shape[0] // 2 + 1
            #print(zero_slope_lines_left[::-1][:x])
            
            ret_level_left = np.median(zero_slope_lines_left[::-1][:x][..., 1])
            #print('Level LEFT: ', ret_level_left)

            ret_bevel_top_left = np.mean(zero_slope_lines_left[::-1][:x][..., 2])
            #print('Bevel Top LEFT: ', ret_bevel_top_left)


        if zero_slope_lines_right:
            zero_slope_lines_right = np.array(zero_slope_lines_right)

            # Sort the lines with length. 
            index = np.argsort(zero_slope_lines_right, axis = 0)
            index_length = index[..., 4]
            zero_slope_lines_right = zero_slope_lines_right[index_length]

            # Get the longest X lines:
            x = zero_slope_lines_right.shape[0] // 2 + 1
            #print(zero_slope_lines_right[::-1][:x])
            
            ret_level_right = np.median(zero_slope_lines_right[::-1][:x][..., 1])
            #print('Level RIGHT: ', ret_level_right)

            ret_bevel_top_right = np.mean(zero_slope_lines_right[::-1][:x][..., 0])
            #print('Bevel Top RIGHT: ', ret_bevel_top_right)

    else:
        print('Failed to found enough surface lines. ')

    return (ret_level_left, ret_level_right, ret_bevel_top_left, ret_bevel_top_right)

"""
＝＝＝＝＝＝＝＝＝＝＝＝把 getSurfaceLevel 分成两部分，用于加速＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
得到焊接表面到底面深度
"""
def getSurfaceLevel_Part_One(image, max_angle = 5, min_length = 200, max_line_gap = 25):
    np.set_printoptions(precision=3, suppress=True)

    lines = getLines(image, min_length=min_length, max_line_gap=max_line_gap)

    return lines, image, max_angle, min_length, max_line_gap

"""
＝＝＝＝＝＝＝＝＝＝＝＝把 getSurfaceLevel 分成两部分，用于加速＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
得到焊接表面到底面深度
"""
def getSurfaceLevel_Part_Two(lines, image, max_angle = 5, min_length = 200, max_line_gap = 25):
    np.set_printoptions(precision=3, suppress=True)
    # Get the dimention of the image and then determine the certer. 
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    ret_level_left = cX
    ret_level_right = cX
    ret_bevel_top_left = -1
    ret_bevel_top_right = -1

    zero_slope_lines_left = []
    zero_slope_lines_right = []
    max_radian = max_angle * np.pi / 180

    #print('No LINE: ', type(lines))

    if type(lines) != type(None) and lines.size > 0:
        for line in lines: 

            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x1 - x2)** 2 + (y1 - y2)** 2)
            if (x2 != x1):
                theta = np.arctan((y2 - y1) / (x2 - x1))
            else:
                theta = 0
            theta_apro = np.around(theta, 1)

            if theta_apro < max_radian and theta_apro > -max_radian:
            
                if (x1 + x2) < w:
                    zero_slope_lines_left.append([x1, y1, x2, y2, length, theta_apro, theta])
                else:
                    zero_slope_lines_right.append([x1, y1, x2, y2, length, theta_apro, theta])

    if zero_slope_lines_left or zero_slope_lines_right:

        reference_lines = []
        ret_level_left = -1
        ret_level_right = -1

        if zero_slope_lines_left:
            zero_slope_lines_left = np.array(zero_slope_lines_left)

            # Sort the lines with length. 
            index = np.argsort(zero_slope_lines_left, axis = 0)
            index_length = index[..., 4]
            zero_slope_lines_left = zero_slope_lines_left[index_length]

            # Get the longest X lines:
            x = zero_slope_lines_left.shape[0] // 2 + 1
            #print(zero_slope_lines_left[::-1][:x])
            
            ret_level_left = np.median(zero_slope_lines_left[::-1][:x][..., 1])
            #print('Level LEFT: ', ret_level_left)

            ret_bevel_top_left = np.mean(zero_slope_lines_left[::-1][:x][..., 2])
            #print('Bevel Top LEFT: ', ret_bevel_top_left)


        if zero_slope_lines_right:
            zero_slope_lines_right = np.array(zero_slope_lines_right)

            # Sort the lines with length. 
            index = np.argsort(zero_slope_lines_right, axis = 0)
            index_length = index[..., 4]
            zero_slope_lines_right = zero_slope_lines_right[index_length]

            # Get the longest X lines:
            x = zero_slope_lines_right.shape[0] // 2 + 1
            #print(zero_slope_lines_right[::-1][:x])
            
            ret_level_right = np.median(zero_slope_lines_right[::-1][:x][..., 1])
            #print('Level RIGHT: ', ret_level_right)

            ret_bevel_top_right = np.mean(zero_slope_lines_right[::-1][:x][..., 0])
            #print('Bevel Top RIGHT: ', ret_bevel_top_right)

    else:
        print('Failed to found enough surface lines. ')

    return (ret_level_left, ret_level_right, ret_bevel_top_left, ret_bevel_top_right)



"""
韩老师的函数＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
输入核心点组成的图像，过滤得到核心线段。
"""
def followCoreLine(lib, image, ref_level, min_gap = 20, black_limit = 0):
    (h, w) = image.shape[:2]

    dst = np.zeros(shape = image.shape, dtype = np.uint8)

    src = np.ctypeslib.as_ctypes(image)
    dst = np.ctypeslib.as_ctypes(dst)

    #dst = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_uint8) * w * h)

    level_left, level_right = ref_level
    level_left = int(level_left)
    level_right = int(level_right)

#    print('h=',h,'w=',w,'level_left=',level_left,'level_right=', level_right,'min_gap=',min_gap, 'black_limit=', black_limit)
    lib.followCoreLine(src, dst, h, w, level_left, level_right, min_gap, black_limit)


    dst = ctypes.cast(dst, ctypes.POINTER(ctypes.c_uint8))
    lineImage = np.ctypeslib.as_array(dst, shape = image.shape)
    #lineImage = dst

#    cv2.imshow('imshow',lineImage)
    return lineImage

"""
韩老师的函数＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
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
韩老师的函数＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
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
韩老师的函数＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
在彩色图像中叠加灰度图像线段。
fill_color: 灰度图像中的点在彩色图像中显示的颜色。
"""
def fill2ColorImage(lib, colorImage, grayImage, fill_color = (255, 0, 0)):
    (h, w) = colorImage.shape[:2]
    colorShape = colorImage.shape

    color = np.ctypeslib.as_ctypes(colorImage)
    gray = np.ctypeslib.as_ctypes(grayImage)

    (r, g, b) = fill_color
    
    lib.fill2ColorImage(color, gray, h, w, 0, r, g, b)

    color = ctypes.cast(color, ctypes.POINTER(ctypes.c_uint8))
    mergedImage = np.ctypeslib.as_array(color, shape = colorShape)

    return mergedImage

"""
韩老师的函数＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
从识别轮廓线得到轮廓数组。轮廓数组长度为图像宽度，每个单元包含该列线段高度。
轮廓数组可以传送给后续函数做焊枪位置识别。
"""
def coreLine2Index(lib, coreImage):
    (h, w) = coreImage.shape[:2]
    
    coreLine = np.ctypeslib.as_ctypes(coreImage)
    index = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_int) * w)
    
    lib.coreLine2Index(coreLine, h, w, index)

    index = ctypes.cast(index, ctypes.POINTER(ctypes.c_int))
    index_array = np.ctypeslib.as_array(index, shape = (w,))

    return index_array

"""
输入轮廓线图像得到焊缝最底部小平台中点位置。
bottom_thick: 底部小平台厚度
noisy_pixels: 作为噪音滤除的像素数目
"""
def getBottomCenter(lib, coreImage, bottom_thick = 30, noisy_pixels = 0):
    index = coreLine2Index(lib, coreImage)
    srt = index.argsort(kind = 'stable')
    idx = srt[:bottom_thick]

    bottom = index[idx]

    level = int(np.mean( bottom[noisy_pixels:(bottom.size - noisy_pixels)] ))
    center = int(np.mean( idx[noisy_pixels:(idx.size - noisy_pixels)] ))
    
    return center, level

"""
输入轮廓线图像平均法自动补足中间缺失的像素。
"""
def fillLineGaps(lib, coreImage, start_pixel = 0):
    (h, w) = coreImage.shape[:2]
    
    coreLine = np.ctypeslib.as_ctypes(coreImage)
    outImage = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_uint8) * w * h)

    lib.fillLineGaps(coreLine, outImage, h, w, start_pixel)

    outImage = ctypes.cast(outImage, ctypes.POINTER(ctypes.c_uint8))
    resImage = np.ctypeslib.as_array(outImage, shape = (h, w))

    return resImage

"""
输入彩色图像，焊缝底部中点位置画出标志线。
"""
def drawTag(image, b_center, b_level):
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

    cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 5)

"""
已知 ROI 区域中 曲线点的坐标值，求距离其弦最远的点的位置。
"""
def Distance_ROI_Fine(First_ROI_ndata):
    # 已知 ROI 区域中 曲线点的坐标值，求距离其弦最远的点的位置。
    # 第一个端点
    Point_ROI_1_X = 0
    Point_ROI_1_Y = First_ROI_ndata[0]
    # 最后一个端点
    Point_ROI_2_X = size(First_ROI_ndata)-1
    Point_ROI_2_Y = First_ROI_ndata[size(First_ROI_ndata)-1]
    # 这两个点连接成的直线

    # 求出所有点到直线的距离
    Distance_1 = np.zeros((size(First_ROI_ndata), 1))

    # 已经知道了曲线的两个端点，求直线方程
    if (Point_ROI_1_X != Point_ROI_2_X):
        k_1_ROI = (Point_ROI_1_Y - Point_ROI_2_Y) / (Point_ROI_1_X - Point_ROI_2_X)
        b_1_ROI = (Point_ROI_1_X * Point_ROI_2_Y - Point_ROI_2_X * Point_ROI_1_Y) / (Point_ROI_1_X - Point_ROI_2_X)
        
        for i in range(size(First_ROI_ndata)):
            Distance_1[i] = abs((First_ROI_ndata[i] - k_1_ROI * i - b_1_ROI) / math.sqrt(1 + k_1_ROI * k_1_ROI))
            Array_Distance_1 = np.array(Distance_1)
        Index_Key_1 = np.argmax(Array_Distance_1)
    else:
        Index_Key_1 = 0

    return Index_Key_1


"""
已知 ROI 区域中 焊缝激光曲线点的坐标值，求两侧拐点的位置
输入：
输出：
"""
def FindKeyPointsWithCoreLineHanDong(result):
    (h, w) = result.shape[:2]
    position = np.zeros((w, 1))
    for i in range(w):
        result_list = result[:, i].tolist()
        #position[0, i] = i
        if 255 in result_list:
            position[i, 0] = result_list.index(255)
        else:
            if i != 0:
                position[i, 0] = position[i - 1, 0]
    
    for i in range(w):
        if position[w - i - 1, 0] == 0:
            position[w - i - 1, 0] = position[w - i, 0]

    # 距离两侧端点最大距离的点的位置索引＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    index = Distance_ROI_Fine(position)

    # 在整个ＲＯＩ区域图中显示出距离最大值的点位置-------------------------
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    #plt.imshow(result)
    #ax.set_title('-------------')
    #plt.plot(index, position[index, 0], '*', 'Color', 'blue')
    #plt.plot(Matrix_Contours[np.argmax(h_P0_Pk)][0, 0], Matrix_Contours[np.argmax(h_P0_Pk)][0, 1], '*', 'Color', 'blue')
    #plt.show()

    # 截取ROI区域最大点的左侧的position
    position_left = position[0:index]
    index_left = Distance_ROI_Fine(position_left)

    # 在最大值点位置的左侧的图中显示出距离该图中曲线端点的最大值的位置索引-------------------------
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    #plt.imshow(result[:,0:index])
    #ax.set_title('-------------')
    #plt.plot(index_left, position_left[index_left, 0], '*', 'Color', 'blue')
    #plt.plot(Matrix_Contours[np.argmax(h_P0_Pk)][0, 0], Matrix_Contours[np.argmax(h_P0_Pk)][0, 1], '*', 'Color', 'blue')
    #plt.show()

    position_right = position[index:w]
    index_right = Distance_ROI_Fine(position_right)
    
    # 在最大值点位置的右侧的图中，显示出距离该图中曲线端点的最大值的位置索引-------------------------
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    #plt.imshow(result)
    #ax.set_title('-------------')
    #plt.plot(index, position[index, 0], '*', 'Color', 'blue')
    #plt.plot(index_left, position_left[index_left, 0], '*', 'Color', 'blue')
    #plt.plot(index_right+ index, position_right[index_right, 0], '*', 'Color', 'blue')
    #plt.plot(Matrix_Contours[np.argmax(h_P0_Pk)][0, 0], Matrix_Contours[np.argmax(h_P0_Pk)][0, 1], '*', 'Color', 'blue')
    #plt.show()
    return index_left, position_left[index_left, 0], index_right + index, position_right[index_right, 0]



"""
已知 ROI 区域中 焊缝激光曲线点的坐标值，求两侧拐点的位置
输入：
输出：
"""
def FindKeyPointsWithCoreLineTIANWEI(lib, cut_img):
    (h, w) = cut_img.shape[:2]
    #================================================================================
    #use C lib
    img2 = cut_img.copy()   
    w0, h0 = cut_img.shape[:2][::-1]
    #(h0, w0) = img.shape[:2]
    src_img = np.ctypeslib.as_ctypes(img2)
    center_line_pt = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_double) * w0)
    lib.extract_center_line(src_img, w0,h0, center_line_pt)
    
    center_line_pt = ctypes.cast(center_line_pt, ctypes.POINTER(ctypes.c_double))
    Index_Max_Chaque_Colomn = np.ctypeslib.as_array(center_line_pt, shape = (w0,1))      
#    #Index_Max_Chaque_Colomn = Index_Max_Chaque_Colomn.T
    #================================================================================
    # 提取线结构光中心
#    Index_Max_Chaque_Colomn = Abstract_Center_Line(cut_img)

    index_Middle_Max = Distance_ROI_Fine(Index_Max_Chaque_Colomn)

    # 在整个ＲＯＩ区域图中显示出距离最大值的点位置-------------------------
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    #plt.imshow(cut_img)
    #ax.set_title('-------------')
    #plt.plot(index_Middle_Max, Index_Max_Chaque_Colomn[index_Middle_Max, 0], '*', 'Color', 'red')
    #plt.plot(Matrix_Contours[np.argmax(h_P0_Pk)][0, 0], Matrix_Contours[np.argmax(h_P0_Pk)][0, 1], '*', 'Color', 'blue')
    #plt.show()

    # 截取ROI区域最大点的左侧的position
    Index_Max_Chaque_Colomn_left = Index_Max_Chaque_Colomn[0:index_Middle_Max]
    index_TianWei_left = Distance_ROI_Fine(Index_Max_Chaque_Colomn_left)

    # 在最大值点位置的左侧的图中显示出距离该图中曲线端点的最大值的位置索引-------------------------
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    #plt.imshow(cut_img[:,0:index_Middle_Max])
    #ax.set_title('-------------')
    #plt.plot(index_TianWei_left, Index_Max_Chaque_Colomn_left[index_TianWei_left, 0], '*', 'Color', 'red')
    #plt.plot(Matrix_Contours[np.argmax(h_P0_Pk)][0, 0], Matrix_Contours[np.argmax(h_P0_Pk)][0, 1], '*', 'Color', 'blue')
    #plt.show()

    Index_Max_Chaque_Colomn_right = Index_Max_Chaque_Colomn[index_Middle_Max:w]
    index_TianWei_right = Distance_ROI_Fine(Index_Max_Chaque_Colomn_right)
    
    # 在最大值点位置的右侧的图中，显示出距离该图中曲线端点的最大值的位置索引-------------------------
    #fig = plt.figure()
    #ax = fig.add_subplot(1, 1, 1)
    #plt.imshow(cut_img)
    #ax.set_title('-------------')
    #plt.plot(index_Middle_Max, Index_Max_Chaque_Colomn[index_Middle_Max, 0], '*', 'Color', 'red')
    #plt.plot(index_TianWei_left, Index_Max_Chaque_Colomn_left[index_TianWei_left, 0], '*', 'Color', 'red')
    #plt.plot(index_TianWei_right + index_Middle_Max, Index_Max_Chaque_Colomn_right[index_TianWei_right, 0], '*', 'Color', 'red')
    #plt.plot(Matrix_Contours[np.argmax(h_P0_Pk)][0, 0], Matrix_Contours[np.argmax(h_P0_Pk)][0, 1], '*', 'Color', 'blue')
    #plt.show()
    return index_TianWei_left, Index_Max_Chaque_Colomn_left[index_TianWei_left, 0], index_TianWei_right + index_Middle_Max, Index_Max_Chaque_Colomn_right[index_TianWei_right, 0]


"""
利用哈弗变化求取两侧的直线。
"""
def GetLineByHoffLine(cut_img):
    #＝＝＝＝＝＝＝＝＝ 首先对ＲＯＩ区域，进行霍夫变化，提取ＲＯＩ中的水平方向的直线＝＝＝＝＝＝＝＝＝＝＝
    blur = cv2.medianBlur(cut_img, 5)

    kernel = np.ones((3,3),np.uint8)
    ret, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    #cv2.namedWindow('closed', cv2.WINDOW_NORMAL)
    #cv2.imshow('closed', closed)
    #cv2.waitKey(0)
    
    # 类似于韩老师的算法中的 getLines 函数　－－－－－－－－
    (height_closed, width_closed) = closed.shape[:2]
    min_length = width_closed // 20
    max_line_gap = 2
    lines = cv2.HoughLinesP(closed, rho = 1, 
                           theta = np.pi/180, 
                           threshold = 100, 
                           minLineLength = min_length, 
                           maxLineGap=max_line_gap)

    if type(lines) != type(None) and lines.size > 0:
        lines = lines[:, 0,:]
    else:
        lines = []

    #lines = lines[:,0,:]


    return lines, width_closed

"""
求取两侧的直线线段的端点坐标。
"""
def FindKeyPointsByHoffLine(lines, width_closed):

    zero_slope_lines_left = []
    zero_slope_lines_right = []
    zero_slope_lines_middle_left = []
    zero_slope_lines_middle_right = []
    max_angle = 15
    max_radian = max_angle * np.pi / 180
    line_longest_left = np.zeros((1, 4))
    line_longest_right = np.zeros((1, 4))
    x_1_left = 0
    y_1_left = 0
    x_2_left = 0
    y_2_left = 0
    x_1_right = 0
    y_1_right = 0
    x_2_right = 0
    y_2_right = 0
    

    #if type(lines) != type(None) and lines.size > 0:
    if len(lines) != 0:
        for line in lines: 

            # 计算出lines中所有线段的长度，倾斜角度
            #x1, y1, x2, y2 = line[0]
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]
            length = np.sqrt((x1 - x2)** 2 + (y1 - y2)** 2)
            if x2 != x1:
                theta = np.arctan((y2 - y1) / (x2 - x1))
            else:
                theta = 0 
            theta_apro = np.around(theta, 1)

            if theta_apro < max_radian and theta_apro > -max_radian:
            
                # 如果线段两端的端点X坐标值，其和小于width_closed，说明这条直线是在图像的左侧。
                # 说明这个鼠标选取的是左侧的ＲＯＩ
                if (x1 + x2) < width_closed:
                    # 左侧的线段放在 zero_slope_lines_left 中。
                    zero_slope_lines_left.append([x1, y1, x2, y2, length, theta_apro, theta])
                else:
                    # 右侧的线段放在zero_slope_lines_right中。
                    zero_slope_lines_right.append([x1, y1, x2, y2, length, theta_apro, theta])

    if zero_slope_lines_left or zero_slope_lines_right:

        #reference_lines = []
        ret_radian_left = None
        ret_radian_right = None
        ret_radian = 0

        # 对于在鼠标左侧提取出来的水平直线段：
        if zero_slope_lines_left:
            zero_slope_lines_left = np.array(zero_slope_lines_left)

            # Sort the lines with length. 按照长度对线段进行排序
            index = np.argsort(zero_slope_lines_left, axis = 0)
            index_length = index[..., 4]
            zero_slope_lines_left = zero_slope_lines_left[index_length]

            # Get the longest X lines:
            x = zero_slope_lines_left.shape[0] // 4 + 1
            #print(zero_slope_lines_left[::-1][:x])

            ret_radian_left = np.mean(zero_slope_lines_left[::-1][:x][..., 6])
            #print('Radian LEFT: ', ret_radian_left)

            # 显示出左侧最长的线段-------------------------------------------
            # # 找出的最长的直线
            line_longest_left = zero_slope_lines_left[::-1][:x]
            
            x_1_left = line_longest_left[0][0]
            y_1_left = line_longest_left[0][1]
            x_2_left = line_longest_left[0][2]
            y_2_left = line_longest_left[0][3]
            
            x_left = [line_longest_left[0][0], line_longest_left[0][2]]
            y_left = [line_longest_left[0][1], line_longest_left[0][3]]

            #plt.imshow(closed)
            
            #for i in range(len(x_left)):
            #    plt.plot(x_left[i], y_left[i], color='r')
            #    plt.scatter(x_left[i], y_left[i], color='b')
            #plt.show()
            
        if zero_slope_lines_right:
            zero_slope_lines_right = np.array(zero_slope_lines_right)

            # Sort the lines with length. 
            index = np.argsort(zero_slope_lines_right, axis = 0)
            index_length = index[..., 4]
            zero_slope_lines_right = zero_slope_lines_right[index_length]

            # Get the longest X lines:
            x = zero_slope_lines_right.shape[0] // 4 + 1
            #print(zero_slope_lines_right[::-1][:x])
            
            ret_radian_right = np.mean(zero_slope_lines_right[::-1][:x][..., 6])
            #print('Radian RIGHT: ', ret_radian_right)

            # 显示出右侧最长的线段-------------------------------------------
            # 找出的最长的直线
            line_longest_right = zero_slope_lines_right[::-1][:x]
            x_1_right = line_longest_right[0][0]
            y_1_right = line_longest_right[0][1]
            x_2_right = line_longest_right[0][2]
            y_2_right = line_longest_right[0][3]
            
            x_right = [line_longest_right[0][0], line_longest_right[0][2]]
            y_right = [line_longest_right[0][1], line_longest_right[0][3]]

            #plt.imshow(closed)
            
            #for i in range(len(x_right)):
            #    plt.plot(x_left[i], y_left[i], color='r')
            #    plt.plot(x_right[i], y_right[i], color='r')
            #    plt.scatter(x_right[i], y_right[i], color='b')
            #    plt.scatter(x_left[i], y_left[i], color='b')
            #plt.show()

    # 输出左边的第二个点和右边的第一个点的坐标值
    return x_2_left, y_2_left, x_1_right, y_1_right


def FindKeyPointsByHoffLine1(cut_img):

    #＝＝＝＝＝＝＝＝＝ 首先对ＲＯＩ区域，进行霍夫变化，提取ＲＯＩ中的水平方向的直线＝＝＝＝＝＝＝＝＝＝＝
    blur = cv2.medianBlur(cut_img, 5)

    kernel = np.ones((3,3),np.uint8)
    ret, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    #cv2.namedWindow('closed', cv2.WINDOW_NORMAL)
    #cv2.imshow('closed', closed)
    #cv2.waitKey(0)
    
    # 类似于韩老师的算法中的 getLines 函数　－－－－－－－－
    (height_closed, width_closed) = closed.shape[:2]
    min_length = width_closed // 20
    max_line_gap = 2
    lines = cv2.HoughLinesP(closed, rho = 1, 
                           theta = np.pi/180, 
                           threshold = 100, 
                           minLineLength = min_length, 
                           maxLineGap=max_line_gap)
    
    if type(lines) != type(None) and lines.size > 0:
        lines = lines[:, 0,:]
    else:
        lines = []
    
    zero_slope_lines_left = []
    zero_slope_lines_right = []
    zero_slope_lines_middle_left = []
    zero_slope_lines_middle_right = []
    max_angle = 30
    max_radian = max_angle * np.pi / 180
    line_longest_left = np.zeros((1, 4))
    line_longest_right = np.zeros((1, 4))
    line_longest_middle_left = np.zeros((1, 4))
    line_longest_middle_right = np.zeros((1, 4))
    x_1_left = 0
    y_1_left = 0
    x_2_left = 0
    y_2_left = 0
    x_1_right = 0
    y_1_right = 0
    x_2_right = 0
    y_2_right = 0
    

#    if type(lines) != type(None) and lines.size > 0:
    if len(lines) != 0:
        for line in lines: 

            # 计算出lines中所有线段的长度，倾斜角度
            #x1, y1, x2, y2 = line[0]
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]
            length = np.sqrt((x1 - x2)** 2 + (y1 - y2)** 2)
            if (x2 != x1):
                theta = np.arctan((y2 - y1) / (x2 - x1))
            else:
                theta = 0 
            theta_apro = np.around(theta, 1)

            if theta_apro < max_radian and theta_apro > -max_radian:
            
                # 如果线段两端的端点X坐标值，其和小于width_closed，说明这条直线是在图像的左侧。
                # 说明这个鼠标选取的是左侧的ＲＯＩ
                if (x1 + x2) < width_closed:
                    # 左侧的线段放在 zero_slope_lines_left 中。
                    zero_slope_lines_left.append([x1, y1, x2, y2, length, theta_apro, theta])
                else:
                    # 右侧的线段放在zero_slope_lines_right中。
                    zero_slope_lines_right.append([x1, y1, x2, y2, length, theta_apro, theta])

    if zero_slope_lines_left or zero_slope_lines_right:

        #reference_lines = []
        ret_radian_left = None
        ret_radian_right = None
        ret_radian = 0

        # 对于在鼠标左侧提取出来的水平直线段：
        if zero_slope_lines_left:
            zero_slope_lines_left = np.array(zero_slope_lines_left)

            # Sort the lines with length. 按照长度对线段进行排序
            index = np.argsort(zero_slope_lines_left, axis = 0)
            index_length = index[..., 4]
            zero_slope_lines_left = zero_slope_lines_left[index_length]

            # Get the longest X lines:
            x = zero_slope_lines_left.shape[0] // 4 + 1
            #print(zero_slope_lines_left[::-1][:x])

            ret_radian_left = np.mean(zero_slope_lines_left[::-1][:x][..., 6])
            #print('Radian LEFT: ', ret_radian_left)

            # 显示出左侧最长的线段-------------------------------------------
            # # 找出的最长的直线
            line_longest_left = zero_slope_lines_left[::-1][:x]
            
            x_1_left = line_longest_left[0][0]
            y_1_left = line_longest_left[0][1]
            x_2_left = line_longest_left[0][2]
            y_2_left = line_longest_left[0][3]
            
            x_left = [line_longest_left[0][0], line_longest_left[0][2]]
            y_left = [line_longest_left[0][1], line_longest_left[0][3]]

            #plt.imshow(closed)
            
            #for i in range(len(x_left)):
            #    plt.plot(x_left[i], y_left[i], color='r')
            #    plt.scatter(x_left[i], y_left[i], color='b')
            #plt.show()
            
        if zero_slope_lines_right:
            zero_slope_lines_right = np.array(zero_slope_lines_right)

            # Sort the lines with length. 
            index = np.argsort(zero_slope_lines_right, axis = 0)
            index_length = index[..., 4]
            zero_slope_lines_right = zero_slope_lines_right[index_length]

            # Get the longest X lines:
            x = zero_slope_lines_right.shape[0] // 4 + 1
            #print(zero_slope_lines_right[::-1][:x])
            
            ret_radian_right = np.mean(zero_slope_lines_right[::-1][:x][..., 6])
            #print('Radian RIGHT: ', ret_radian_right)

            # 显示出右侧最长的线段-------------------------------------------
            # 找出的最长的直线
            line_longest_right = zero_slope_lines_right[::-1][:x]
            x_1_right = line_longest_right[0][0]
            y_1_right = line_longest_right[0][1]
            x_2_right = line_longest_right[0][2]
            y_2_right = line_longest_right[0][3]
            
            x_right = [line_longest_right[0][0], line_longest_right[0][2]]
            y_right = [line_longest_right[0][1], line_longest_right[0][3]]

    # 当左右两侧都能找到直线的时候;
    if len(zero_slope_lines_left)*len(zero_slope_lines_right):
        
        # 利用所求得的两条直线的相近的两个端点，将图像的中间部分，焊缝坡口部分切出来
        # 焊缝坡口靠左，四分之一坡口宽度的图像为：

        #＝＝＝＝ 修改错误＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        # 错误的原因是：有时找到的右侧的直线的第一个端点的坐标值　x_1_right 　小于　 x_2_left，
        # 导致无法取得图像。
        if (type(zero_slope_lines_left) != type(None)) and (len(zero_slope_lines_left) > 0) and ((x_1_right - x_2_left) > 20):

            cut_img_middle_left = blur[0:height_closed, int(x_2_left):int((x_1_right + 3 * x_2_left) / 4)]

            kernel = np.ones((3, 3), np.uint8)
            ret, cut_img_middle_left_binary = cv2.threshold(cut_img_middle_left, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
            cut_img_middle_left_closed = cv2.morphologyEx(cut_img_middle_left_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # 焊缝坡口靠左，从图像中找到最长的直线
            (height_cut_img_middle_left_closed, width_cut_img_middle_left_closed) = cut_img_middle_left_closed.shape[:2]
            min_length = width_cut_img_middle_left_closed // 40
            max_line_gap = 20
            lines_middle_left = cv2.HoughLinesP(cut_img_middle_left_closed, rho = 1, 
                           theta = np.pi/180, 
                           threshold = 5, 
                           minLineLength = min_length, 
                           maxLineGap=max_line_gap)
            zero_slope_lines_middle_left = []
            max_angle = 75
            max_radian = max_angle * np.pi / 180
            
            if type(lines_middle_left) != type(None) and lines_middle_left.size > 0:
                
                for line in lines_middle_left:
                    
                    # 计算出lineslines_middle_left 中所有线段的长度，倾斜角度
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x1 - x2)** 2 + (y1 - y2)** 2)
                    if (x2 != x1):
                        theta = np.arctan((y2 - y1) / (x2 - x1))
                    else:
                        theta = 0
                    theta_apro = np.around(theta, 1)
                    
                    if theta_apro < max_radian and theta_apro > -max_radian:
                        zero_slope_lines_middle_left.append([x1, y1, x2, y2, length, theta_apro, theta])
                        
                # 如果　zero_slope_lines_middle_left　不为空的话
                if zero_slope_lines_middle_left:
                    #reference_lines = []
                    ret_radian_left = None
                    ret_radian_right = None
                    ret_radian = 0
                    
                    zero_slope_lines_middle_left = np.array(zero_slope_lines_middle_left)
                    # Sort the lines with length. 按照长度对线段进行排序
                    index = np.argsort(zero_slope_lines_middle_left, axis=0)
                    index_length = index[..., 4]
                    zero_slope_lines_middle_left = zero_slope_lines_middle_left[index_length]
                    # Get the longest X lines:
                    x = zero_slope_lines_middle_left.shape[0] // 4 + 1
                    #print(zero_slope_lines_middle_left[::-1][:x])
                    
                    ret_radian_left = np.mean(zero_slope_lines_middle_left[::-1][:x][..., 6])
                    #print('Radian LEFT: ', ret_radian_left)
                    
                    # 显示出左侧最长的线段-------------------------------------------
                    # # 找出的最长的直线
                    line_longest_middle_left = zero_slope_lines_middle_left[::-1][:x]
                    x_1_middle_left = line_longest_middle_left[0][0]
                    y_1_middle_left = line_longest_middle_left[0][1]
                    x_2_middle_left = line_longest_middle_left[0][2]
                    y_2_middle_left = line_longest_middle_left[0][3]
                    
                    x_middle_left = [line_longest_middle_left[0][0], line_longest_middle_left[0][2]]
                    y_middle_left = [line_longest_middle_left[0][1], line_longest_middle_left[0][3]]
            

        # 焊缝坡口靠右侧，四分之一坡口宽度的图像为：
        if type(zero_slope_lines_right) != type(None) and len(zero_slope_lines_right) > 0 and ((x_1_right - x_2_left) > 20):
            # 焊缝坡口靠左，四分之一坡口宽度的图像为：
            cut_img_middle_right = blur[0:height_closed, int((x_2_left + 3*x_1_right)/4):int(x_1_right )]
            
            kernel = np.ones((3, 3), np.uint8)
            ret, cut_img_middle_right_binary = cv2.threshold(cut_img_middle_right, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            cut_img_middle_right_closed = cv2.morphologyEx(cut_img_middle_right_binary, cv2.MORPH_CLOSE, kernel, iterations=1)
            
            # 焊缝坡口靠左，从图像中找到最长的直线
            (height_cut_img_middle_right_closed, width_cut_img_middle_right_closed) = cut_img_middle_right_closed.shape[:2]
            min_length = width_cut_img_middle_right_closed // 40
            max_line_gap = 20
            lines_middle_right = cv2.HoughLinesP(cut_img_middle_right_closed, rho = 1, 
                           theta = np.pi/180, 
                           threshold = 5, 
                           minLineLength = min_length, 
                           maxLineGap=max_line_gap)
            zero_slope_lines_middle_right = []
            max_angle = 75
            max_radian = max_angle * np.pi / 180
            
            if type(lines_middle_right) != type(None) and lines_middle_right.size > 0:
                
                for line in lines_middle_right:
                    
                    # 计算出lineslines_middle_right 中所有线段的长度，倾斜角度
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x1 - x2)** 2 + (y1 - y2)** 2)
                    if (x2 != x1):
                        theta = np.arctan((y2 - y1) / (x2 - x1))
                    else:
                        theta = 0
                    theta_apro = np.around(theta, 1)
                    
                    if theta_apro < max_radian and theta_apro > -max_radian:
                        zero_slope_lines_middle_right.append([x1, y1, x2, y2, length, theta_apro, theta])
                        
                # 如果　zero_slope_lines_middle_right　不为空的话
                if zero_slope_lines_middle_right:
                    #reference_lines = []
                    #ret_radian_left = None
                    ret_radian_right = None
                    ret_radian = 0
                    
                    zero_slope_lines_middle_right = np.array(zero_slope_lines_middle_right)
                    # Sort the lines with length. 按照长度对线段进行排序
                    index = np.argsort(zero_slope_lines_middle_right, axis=0)
                    index_length = index[..., 4]
                    zero_slope_lines_middle_right = zero_slope_lines_middle_right[index_length]
                    # Get the longest X lines:
                    x = zero_slope_lines_middle_right.shape[0] // 4 + 1
                    #print(zero_slope_lines_middle_right[::-1][:x])
                    
                    ret_radian_right = np.mean(zero_slope_lines_middle_right[::-1][:x][..., 6])
                    #print('Radian LEFT: ', ret_radian_right)
                    
                    # 显示出右侧最长的线段-------------------------------------------
                    # # 找出的最长的直线
                    line_longest_middle_right = zero_slope_lines_middle_right[::-1][:x]
                    x_1_middle_right = line_longest_middle_right[0][0]
                    y_1_middle_right = line_longest_middle_right[0][1]
                    x_2_middle_right = line_longest_middle_right[0][2]
                    y_2_middle_right = line_longest_middle_right[0][3]
                    
                    x_middle_right = [line_longest_middle_right[0][0], line_longest_middle_right[0][2]]
                    y_middle_right = [line_longest_middle_right[0][1], line_longest_middle_right[0][3]]
            
                    
    
    # 得出的左右两侧焊缝拐点附近的４条直线的端点
    if len(zero_slope_lines_middle_left) != 0 and len(zero_slope_lines_middle_right) != 0:
        x = [line_longest_left[0][0], line_longest_left[0][2],
                    line_longest_right[0][0], line_longest_right[0][2],
                    line_longest_middle_left[0][0] + x_2_left, line_longest_middle_left[0][2] +x_2_left,
                    line_longest_middle_right[0][0] + int((x_2_left + 3*x_1_right)/4), line_longest_middle_right[0][2] + int((x_2_left + 3*x_1_right)/4)]
        y = [line_longest_left[0][1], line_longest_left[0][3],
                     line_longest_right[0][1], line_longest_right[0][3],
                     line_longest_middle_left[0][1], line_longest_middle_left[0][3],
                     line_longest_middle_right[0][1], line_longest_middle_right[0][3]]
    elif len(zero_slope_lines_middle_left) == 0 or len(zero_slope_lines_middle_right) == 0:
        x = [line_longest_left[0][0], line_longest_left[0][2],
                    line_longest_right[0][0], line_longest_right[0][2],
                    x_2_left, x_2_left,
                    x_1_right, x_1_right]
        y = [line_longest_left[0][1], line_longest_left[0][3],
                     line_longest_right[0][1], line_longest_right[0][3],
                     y_2_left, y_2_left, y_1_right, y_1_right]


    
    

    #print('x0 - x7 ', x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7])
    #print('y0 - y7 ', y[0],y[1],y[2],y[3],y[4],y[5],y[6],y[7])

    
    # 分别求出两条直线的交点
    if (x[5] != x[4]) and (x[1] != x[0]) and (y[5] != y[4]) and (y[1] != y[0]) and (x[7] != x[6]) and (x[3] != x[2]) and (y[7] != y[6]) and (y[3] != y[2]) and (((y[3] - y[2]) / (x[3] - x[2]) - (y[7] - y[6]) / (x[7] - x[6])) != 0) and (((x[3] - x[2]) / (y[3] - y[2]) - (x[7] - x[6]) / (y[7] - y[6])) != 0) :
        x_Key_left = (y[4] - y[0] - x[4] * (y[5] - y[4]) / (x[5] - x[4]) + x[0] * (y[1] - y[0]) / (x[1] - x[0])) / ((y[1] - y[0]) / (x[1] - x[0]) - (y[5] - y[4]) / (x[5] - x[4]))
        y_Key_left = (x[4] - x[0] - y[4] * (x[5] - x[4]) / (y[5] - y[4]) + y[0] * (x[1] - x[0]) / (y[1] - y[0])) / ((x[1] - x[0]) / (y[1] - y[0]) - (x[5] - x[4]) / (y[5] - y[4]))
        x_Key_right = (y[6] - y[2] - x[6] * (y[7] - y[6]) / (x[7] - x[6]) + x[2] * (y[3] - y[2]) / (x[3] - x[2])) / ((y[3] - y[2]) / (x[3] - x[2]) - (y[7] - y[6]) / (x[7] - x[6]))
        y_Key_right = (x[6] - x[2] - y[6] * (x[7] - x[6]) / (y[7] - y[6]) + y[2] * (x[3] - x[2]) / (y[3] - y[2])) / ((x[3] - x[2]) / (y[3] - y[2]) - (x[7] - x[6]) / (y[7] - y[6]))

        x_Points_Key = [x_Key_left, x_Key_right]
        y_Points_Key = [y_Key_left, y_Key_right]
    
    # 当左右两侧的直线的端点的纵坐标相等时：
    else:
        if ((y[2]==y[3]) and (y[0] != y[1])) and (x[5] != x[4]) and (y[5] != y[4]) and (x[1] != x[0]) and (y[7] != y[6]):
            x_Key_left = (y[4] - y[0] - x[4] * (y[5] - y[4]) / (x[5] - x[4]) + x[0] * (y[1] - y[0]) / (x[1] - x[0])) / ((y[1] - y[0]) / (x[1] - x[0]) - (y[5] - y[4]) / (x[5] - x[4]))
            y_Key_left = (x[4] - x[0] - y[4] * (x[5] - x[4]) / (y[5] - y[4]) + y[0] * (x[1] - x[0]) / (y[1] - y[0])) / ((x[1] - x[0]) / (y[1] - y[0]) - (x[5] - x[4]) / (y[5] - y[4]))
            x_Key_right = x[6]+(y[3]-y[6])*(x[7]-x[6])/(y[7]-y[6])
            y_Key_right = y[2]
        elif ((y[2]!=y[3]) and (y[0] == y[1])) and (x[5] != x[4]) and (y[5] != y[4]) and (x[7] != x[6]) and (x[3] != x[2]) and (y[7] != y[6]):
            x_Key_left = x[4]+(y[1]-y[4])*(x[5]-x[4])/(y[5]-y[4])
            y_Key_left = y[0]
            x_Key_right = (y[6] - y[2] - x[6] * (y[7] - y[6]) / (x[7] - x[6]) + x[2] * (y[3] - y[2]) / (x[3] - x[2])) / ((y[3] - y[2]) / (x[3] - x[2]) - (y[7] - y[6]) / (x[7] - x[6]))
            y_Key_right = (x[6] - x[2] - y[6] * (x[7] - x[6]) / (y[7] - y[6]) + y[2] * (x[3] - x[2]) / (y[3] - y[2])) / ((x[3] - x[2]) / (y[3] - y[2]) - (x[7] - x[6]) / (y[7] - y[6]))
        elif ((y[2] == y[3]) and (y[0] == y[1])) and (y[5] != y[4]) and (y[7] != y[6]):
            x_Key_left = x[4]+(y[1]-y[4])*(x[5]-x[4])/(y[5]-y[4])
            y_Key_left = y[0]
            x_Key_right = x[6]+(y[3]-y[6])*(x[7]-x[6])/(y[7]-y[6])
            y_Key_right = y[2]
        else:
            x_Key_left = x[1]
            y_Key_left = y[1]
            x_Key_right = x[2]
            y_Key_right = y[2]

        x_Points_Key = [x_Key_left, x_Key_right]
        y_Points_Key = [y_Key_left, y_Key_right]

    return x_Key_left, y_Key_left, x_Key_right, y_Key_right


"""
利用哈弗变化求取两侧的直线线段的端点坐标。
"""
def FindHorizonalLineByHoff(lines):

     #＝＝＝＝＝＝＝＝＝ 首先对ＲＯＩ区域，进行霍夫变化，提取ＲＯＩ中的水平方向的直线＝＝＝＝＝＝＝＝＝＝＝
    #blur = cv2.medianBlur(cut_img, 5)

    #kernel = np.ones((3,3),np.uint8)
    #ret, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 求出该图像的灰度平均值，用于判断哪边的图像更暗一些。
    # 更暗的图像说明其打磨的反光要小一些。
    #Mean = mean(closed)

    #cv2.namedWindow('closed', cv2.WINDOW_NORMAL)
    #cv2.imshow('closed', closed)
    #cv2.waitKey(0)
    
    # 类似于韩老师的算法中的 getLines 函数　－－－－－－－－
    #(height_closed, width_closed) = closed.shape[:2]
    #min_length = width_closed // 20
    #max_line_gap = 2
    #lines = cv2.HoughLinesP(closed, rho = 1, 
    #                       theta = np.pi/180, 
    #                       threshold = 100, 
    #                       minLineLength = min_length, 
    #                       maxLineGap=max_line_gap)
    
    zero_slope_lines = []
    zero_slope_lines_right = []
    zero_slope_lines_middle_left = []
    zero_slope_lines_middle_right = []
    max_angle = 60
    max_radian = max_angle * np.pi / 180
    line_longest = np.zeros((1, 4))
    x_1 = 0
    y_1 = 0
    x_2 = 0
    y_2 = 0

    if type(lines) != type(None) and lines.size > 0:
        for line in lines: 

            # 计算出lines中所有线段的长度，倾斜角度
            #x1, y1, x2, y2 = line[0]
            x1 = line[0]
            y1 = line[1]
            x2 = line[2]
            y2 = line[3]
            length = np.sqrt((x1 - x2)** 2 + (y1 - y2)** 2)
            if (x2 != x1):
                theta = np.arctan((y2 - y1) / (x2 - x1))
            else:
                theta = 0
            theta_apro = np.around(theta, 1)

            if theta_apro < max_radian and theta_apro > -max_radian:
                
                zero_slope_lines.append([x1, y1, x2, y2, length, theta_apro, theta])
                

    if zero_slope_lines:
        
        zero_slope_lines = np.array(zero_slope_lines)
        
        # Sort the lines with length. 按照长度对线段进行排序
        index = np.argsort(zero_slope_lines, axis = 0)
        index_length = index[..., 4]
        zero_slope_lines = zero_slope_lines[index_length]

        # Get the longest X lines:
        x = zero_slope_lines.shape[0] // 4 + 1
        #print(zero_slope_lines[::-1][:x])

        # # 找出的最长的直线
        line_longest = zero_slope_lines[::-1][:x]
            
        x_1 = line_longest[0][0]
        y_1 = line_longest[0][1]
        x_2 = line_longest[0][2]
        y_2 = line_longest[0][3]
            
        x = [line_longest[0][0], line_longest[0][2]]
        y = [line_longest[0][1], line_longest[0][3]]

        #plt.imshow(closed)
            
        #for i in range(len(x)):
        #    plt.plot(x[i], y[i], color='r')
        #    plt.scatter(x[i], y[i], color='b')
        #plt.show()

    #return Mean, x_1, y_1, x_2, y_2
    return x_1, y_1, x_2, y_2


def FindKeyPointsByHoffLine2(lib_FYF, cut_img):
    (height, width) = cut_img.shape[:2]
    #Mean_left = 0
    x_1_left = 0
    y_1_left = 0
    x_2_left = 0
    y_2_left = 0


    #==========================将图像切成左右两个部分＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    #==============　左侧部分　＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    cut_img_left = cut_img[0:height, 0:int(np.round(width / 2))]

    Lines_Left, width_Left = GetLineByHoffLine(cut_img_left)

    #Mean_left, x_1_left, y_1_left, x_2_left, y_2_left = FindHorizonalLineByHoff(Lines_Left)
#    x_1_left, y_1_left, x_2_left, y_2_left = FindHorizonalLineByHoff(Lines_Left)
    
    #=================================================================================================
    # =============== 增加判断机制　＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    if len(Lines_Left) != 0:
        # C code
        (h, w) = Lines_Left.shape[:2]
        src1 = np.ctypeslib.as_ctypes(Lines_Left)
        out1 = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_int) * 4)
        lib_FYF.FindHorizonalLineByHoff(src1, w, h, out1)
        out1 = ctypes.cast(out1, ctypes.POINTER(ctypes.c_int))
        out1_array = np.ctypeslib.as_array(out1, shape=(4, 1))
        x_1_left = out1_array[0]
        y_1_left = out1_array[1]
        x_2_left = out1_array[2]
        y_2_left = out1_array[3]
    else:
        x_1_left = 0
        y_1_left = 0
        x_2_left = 0
        y_2_left = 0

   
    #=================================================================================================
    #==============　右侧部分　＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    cut_img_right = cut_img[0:height, int(np.round(width / 2)):width]

    Lines_Right, width_Right = GetLineByHoffLine(cut_img_right)

    #Mean_right = 0
    x_1_right = 0
    y_1_right = 0
    x_2_right = 0
    y_2_right = 0

    #Mean_right, x_1_right, y_1_right, x_2_right, y_2_right = FindHorizonalLineByHoff(Lines_Right)
#    x_1_right, y_1_right, x_2_right, y_2_right = FindHorizonalLineByHoff(Lines_Right)
    #=================================================================================================
    # =============== 增加判断机制　＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    if len(Lines_Right) != 0:
        # C code
        (h, w) = Lines_Right.shape[:2]
        src2 = np.ctypeslib.as_ctypes(Lines_Right)
        out2 = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_int) * 4)
        lib_FYF.FindHorizonalLineByHoff(src2, w, h, out2)
        out2 = ctypes.cast(out2, ctypes.POINTER(ctypes.c_int))
        out2_array = np.ctypeslib.as_array(out2, shape=(4, 1))
        x_1_right = out2_array[0]
        y_1_right = out2_array[1]
        x_2_right = out2_array[2]
        y_2_right = out2_array[3]
    else:
        x_1_right = 0
        y_1_right = 0
        x_2_right = 0
        y_2_right = 0
    

    
    #return Mean_left, Mean_right, x_2_left, y_2_left, (x_1_right + width / 2), y_1_right
    return x_2_left, y_2_left, (x_1_right + width / 2), y_1_right

"""
求出最相邻的点，排除掉超过阈值范围的点
"""
def AbandonPointsByDistance(X_Left, Y_Left, X_Right, Y_Right):
    Mean_X_Left = get_average(X_Left)
    Mean_Y_Left = get_average(Y_Left)
    Mean_X_Right = get_average(X_Right)
    Mean_Y_Right = get_average(Y_Right)

    X_Left_Available = []
    Y_Left_Available = []
    X_Right_Available = []
    Y_Right_Available = []

    for i in range(len(X_Left)):
        for j in range((len(X_Left))):
            if (i != j) and (math.sqrt((X_Left[i] - X_Left[j])** 2 + (Y_Left[i] - Y_Left[j])** 2) < 20):
                X_Left_Available.append(X_Left[i])
                X_Left_Available.append(X_Left[j])
                Y_Left_Available.append(Y_Left[i])
                Y_Left_Available.append(Y_Left[j])
                
    for i in range(len(X_Right)):
        for j in range((len(X_Right))):
            if (i != j) and (math.sqrt((X_Right[i] - X_Right[j])** 2 + (Y_Right[i] - Y_Right[j])** 2) < 20): 
                X_Right_Available.append(X_Right[i])
                X_Right_Available.append(X_Right[j])
                Y_Right_Available.append(Y_Right[i])
                Y_Right_Available.append(Y_Right[j])

    if len(X_Left_Available) != 0:
        Mean_X_Left = get_average(X_Left_Available)
    if len(Y_Left_Available) != 0:
        Mean_Y_Left = get_average(Y_Left_Available)
    if len(X_Right_Available) != 0:
        Mean_X_Right = get_average(X_Right_Available)
    if len(Y_Right_Available) != 0:
        Mean_Y_Right = get_average(Y_Right_Available)

    return Mean_X_Left, Mean_Y_Left, Mean_X_Right, Mean_Y_Right




            
        
def main():
    global img
    global value
    global Distance_Two_Points
    global Distance_Center_ROI_left
    global Distance_Center_ROI_right
    global Distance_Center_ROI_up
    global Distance_Center_ROI_down
    global Distance_Center_Heightest_Line

    # 读取视频文件。
    # ============== Not Available ===================
    #VideoName = '横焊脉冲填充0.3%衰减10000曝光16mm镜头从左到右2.avi'
    #VideoName = '立焊0.3%衰减片10000曝光16mm镜头光圈不是最大聚焦在焊缝底部和表面之间表面.avi'
    #VideoName = '填充_未焊接_0.3%衰减_16mm镜头_10000曝光(1).avi'

    

    # ==============ok===================
    
#    VideoName = 'F:\博清科技\weld_video\立焊0.3%衰减10000曝光16mm镜头12mm钢板盖面.avi'      #hard
#    VideoName = 'F:\博清科技\weld_video\横焊打底0.3%衰减片10000曝光16mm镜头从左到右.avi'
#    VideoName = 'F:\博清科技\weld_video\盖面_焊接_0.3%衰减_16mm镜头_10000曝光.avi'
#    VideoName = 'F:\博清科技\weld_video\output190918.avi'
#    VideoName = 'F:\博清科技\weld_video\横焊脉冲盖面0.3%衰减10000曝光16mm镜头从左到右.avi'
#    VideoName = 'F:\博清科技\weld_video\横焊脉冲填充0.3%衰减10000曝光16mm镜头从左到右2.avi'      #hard
#    VideoName = 'F:\博清科技\weld_video\横焊脉冲填充0.3%衰减10000曝光16mm镜头从左到右.avi'
#    VideoName = 'F:\博清科技\weld_video\填充_未焊接_0.3%衰减_16mm镜头_10000曝光.avi'            #hard
#    VideoName = 'F:\博清科技\weld_video\立焊0.3%衰减650nm滤光10000曝光多钉点1.avi'
#    VideoName = 'F:\博清科技\weld_video\立焊0.3%衰减650nm滤光10000曝光多钉点2.avi'
#    VideoName = 'F:\博清科技\weld_video\立焊0.3%衰减650nm滤光10000曝光多钉点3.avi'
#    VideoName = 'F:\博清科技\weld_video\立焊0.3%衰减650nm滤光10000曝光多钉点4.avi'
#    VideoName = 'F:\博清科技\weld_video\横焊脉冲填充0.3%衰减10000曝光16mm镜头从左到右.avi'
#    VideoName = 'F:\博清科技\weld_video\立焊0.3%衰减片10000曝光16mm镜头最大光圈聚焦在焊缝表面.avi'
#    VideoName = 'F:\博清科技\weld_video\立焊0.3%衰减片10000曝光16mm镜头光圈不是最大聚焦在焊缝底部和表面之间表面.avi'       #hard
#    VideoName = 'F:\博清科技\weld_video\立焊打底(有挡板)650nm滤光0.3%衰减16mm镜头180mm距离2.avi'
#    VideoName = 'F:\博清科技\weld_video\立焊打底(有挡板)650nm滤光0.3%衰减16mm镜头180mm距离.avi'
#    VideoName = 'F:\博清科技\weld_video\立焊0.3衰减片10000曝光16mm镜头立向上.avi'
#    VideoName = 'F:\博清科技\weld_video\填充_未焊接_0.3%衰减_16mm镜头_10000曝光.avi'
#    VideoName = 'F:\博清科技\weld_video\填充_未焊接_0.3%衰减_16mm镜头_10000曝光.avi'
#    VideoName = 'F:\博清科技\weld_video\填充_未焊接_0.3%衰减_16mm镜头2_10000曝光.avi'
#    VideoName = 'F:\博清科技\weld_video\填充_未焊接_0.3%衰减_16mm镜头3_10000曝光.avi'
#    VideoName = 'F:\博清科技\weld_video\填充_未焊接_0.3%衰减_16mm镜头4_10000曝光.avi'
#    VideoName = 'F:\博清科技\weld_video\填充_未焊接_0.3%衰减_16mm镜头5_10000曝光.avi'
    
#    VideoName = 'F:\博清科技\weld_video\立焊650nm滤光0.3%衰减16mm镜头180mm距离.avi'
    #VideoName = 'D:\weld_video\立焊0.3%衰减650nm滤光10000曝光多钉点1.avi'
    #VideoName = '0.3%Suai Jian 650nm Lv Guang 10000 Bao Guang Duo Ding Dian 3.avi'
    #VideoName = '立焊0.3%衰减片10000曝光16mm镜头最大光圈聚焦在焊缝表面.avi'
#    VideoName = '横焊脉冲填充0.3%衰减10000曝光16mm镜头从左到右.avi'
    #VideoName = '横焊脉冲盖面0.3%衰减10000曝光16mm镜头从左到右.avi'
    #VideoName = '横焊打底0.3%衰减片10000曝光16mm镜头从左到右.avi'
    # 
    #VideoName = '0.3%衰减 10000曝光 16mm镜头 12mm钢板盖面.avi'
    #VideoName = '16mm Jing Tou 660nm Lv Guang Pian.avi'
    #VideoName = 'Heng Han 650nm 16mm 180mm 0.3%.avi'
    #VideoName = 'Heng Han 650nm 16mm 180mm 0.3%2.avi'
    #VideoName = 'Heng Han 650nm 16mm 180mm 0.3% Jing Zhi Zhuang Tai.avi'

    

    #VideoName = 'H100_F6mm_P0.01_650nm_S300.avi'
    #VideoName = 'H100_F6mm_P0.01_650nm_S80.avi'
    #VideoName = 'Li Han Da Di 650nm Lv guang 0.3% Suai jian 16mm Jing Tou 180mm Ju Li 2.avi'
    #VideoName = '650nm Lv Guang 0.3% Suai Jian 16mm Jing Tou 180mm Ju Li.avi'
    #VideoName = '0.3%SuaiJian 650nmLvGuang 10000BaoGuang DuoDingDian2.avi'
    #VideoName = '0.3%Suai Jian 650nm Lv Guang 10000 Bao Guang Duo Ding Dian 1.avi'
    
    #VideoName = 'Li Han Da Di You Dang Ban 650nm Lv Guang 0.3% Suai Jian 16mm Jing Tou 180mm Ju Li.avi'
    #VideoName = 'Li Han Da Di 650nm Lv guang 0.3% Suai jian 16mm Jing Tou 180mm Ju Li 2.avi'
    #vid = cv2.VideoCapture(" 650nm 16mm 180mm 0.3%.avi")
    #vid = cv2.VideoCapture(VideoName)

    #读取摄像头，并设置摄像头的分辨率
    #vid = cv2.VideoCapture(0)
    #vid.set(3, 2048)
    #vid.set(4, 1536)

    #if not vid.isOpened(): # 如果视频没有打开成功的话，
    #    raise IOError("Couldn't open webcam or video:{}".format(input))
    
    #VideoName = '0.3%Suai Jian 650nm Lv Guang 10000 Bao Guang Duo Ding Dian 3.avi'
    VideoName = '/home/dhan/myprog/ws/wsdata/1920x1200/0.3%滤光片10000曝光0增益16mm镜头立向上.avi'

    from frame_provider import frame_provider
    if VideoName is None: 
        ####################################################
        # 打开摄像头
        fp = frame_provider(mode = 'cam', cam_ip = "192.168.40.3")
        ####################################################
    else:
        ####################################################
        # 打开文件
        fp = frame_provider(mode = 'file', file = VideoName)
        ####################################################

    # 开始文件读取或者相机采集进程。判断是否启动成功。
    ok = fp.start()
    if not ok: 
        print("frame_provider初始化错误。")
        return 


    print("=== Start the WS detecting ===")

    print('Load C lib. ')
    #so_file = './ws_c.dll'
    #lib_HD = ctypes.cdll.LoadLibrary(so_file)

    #so_file = './ws_c.dll'
    so_file = './libws_c_20190917.so'
    lib_HD = ctypes.cdll.LoadLibrary(so_file)

    # 增加了Ｃ语言加速部分＝＝＝＝＝＝＝＝＝＝＝
#    lib_FYF = ctypes.cdll.LoadLibrary('./func_win.dll')
    #lib_FYF = ctypes.cdll.LoadLibrary('./func_f_win.dll')
    lib_FYF = ctypes.cdll.LoadLibrary('./func_20190922.so')


#    lib_HD.testlib()

#    kernel = np.ones((5, 5), np.uint8)
    
    # 为normalizeCenter准备数据数组。
#    center_array = []

    # 插入ＲＯＩ选取＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    # cut_img 是用鼠标选取的 ROI 区域图像
    #ret, frame = vid.read()
    frame = fp.read()
    if frame is None:
        print("读取帧出错或文件到达末尾。")
        return 

    #然后对彩色图像进行灰度变换
    Image_Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img = Image_Gray

    #==========================================================================
    # 提升图像对比度
    Value = 5
    #img = Adjust_Image(Image_Gray, Value)
    #==========================================================================
    (h, w) = img.shape[:2]
    src = np.ctypeslib.as_ctypes(img)
    dst = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_uint8) * w * h)   
    lib_FYF.enhance_img_contrast(src, w, h, Value, dst)
    dst = ctypes.cast(dst, ctypes.POINTER(ctypes.c_uint8))
    img = np.ctypeslib.as_array(dst, shape = img.shape)
    

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)

    # 鼠标的最小坐标值和ＲＯＩ的宽度，高度
    min_x = min(point1[0], point2[0])
    min_y = min(point1[1], point2[1])
    width = abs(point1[0] - point2[0])
    height = abs(point1[1] - point2[1])

    # 鼠标截取的ＲＯＩ的区域内图像
    cut_img = img[min_y:min_y + height, min_x:min_x + width]

    
    

    # 首先对ＲＯＩ区域内的图像进行获取两侧拐点的运算＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    #=====================利用韩东的函数来求激光中心线的位置==========================================
    filt = cut_img
    mean = filt.mean()
    #black_limit = (int)(mean * 8)
    black_limit = (int)(mean * 2)
    if black_limit > 245:
        black_limit = 245

    if black_limit < 3:
        black_limit = 3

    coreline = getLineImage(lib_HD, filt, black_limit=black_limit, correct_angle=True)
    
    gaps = fillLineGaps(lib_HD, coreline, start_pixel=5)

    result = gaps + coreline

    # 显示出找到的　中心线的位置图像。
    #cv2.namedWindow('result', cv2.WINDOW_NORMAL)
    #cv2.imshow('result', result)
    #cv2.waitKey(0)
    
    #==============方法１========通过激光中心线的位置来求取距离该曲线端点距离最远点的位置===============================================
    # 将韩东的 result　结果转换成　激光线上各点的坐标值＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    # X_0, Y_0 表示激光线上最高点的坐标值＝＝＝＝＝＝＝＝＝＝＝＝＝
    #X_0, Y_0, X_Left_Method_0, Y_Left_Method_0, X_Right_Method_0, Y_Right_Method_0 = FindKeyPointsWithCoreLineHanDong(result)

    #==============方法１========通过激光中心线的位置来求取距离该曲线端点距离最远点的位置===============================================
    # 将韩东的 result　结果转换成　激光线上各点的坐标值＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    # X_Left_Method_0, Y_Left_Method_0, X_Right_Method_0, Y_Right_Method_0 = FindKeyPointsWithCoreLineHanDong(result)
    #=================================================================================================================
    # c code
    (h, w) = result.shape[:2]
    src = np.ctypeslib.as_ctypes(result)
    out = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_int)*4)
    lib_FYF.FindKeyPointsWithCoreLine1(src, w, h, out)
    out = ctypes.cast(out, ctypes.POINTER(ctypes.c_int)) 
    out_array = np.ctypeslib.as_array(out, shape = (4,1))
    X_Left_Method_0 = out_array[0]
    Y_Left_Method_0 = out_array[1]
    X_Right_Method_0 = out_array[2]
    Y_Right_Method_0 = out_array[3]
    #=================================================================================================================
    
    #==============方法2=========利用我以前的提取中心线的函数===================================================
    X_Left_Method_1, Y_Left_Method_1, X_Right_Method_1, Y_Right_Method_1 = FindKeyPointsWithCoreLineTIANWEI(lib_FYF, cut_img)

    #==============方法3=========== 利用哈弗变换求两侧的直线端的端点＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    X_Left_Method_2, Y_Left_Method_2, X_Right_Method_2, Y_Right_Method_2 = FindKeyPointsByHoffLine1(cut_img) 
    
#    LinesByHoffLine, width_closed = GetLineByHoffLine(cut_img)
#    X_Left_Method_2, Y_Left_Method_2, X_Right_Method_2, Y_Right_Method_2 = FindKeyPointsByHoffLine(LinesByHoffLine, width_closed)
    
    #==============================================
    #c code
#    if len(LinesByHoffLine) != 0:
#        (h_lbhl,w_lbhl) = LinesByHoffLine.shape[:2]
#        lines1 = np.ctypeslib.as_ctypes(LinesByHoffLine);
#        out1 = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_int)*4)
#        lib_FYF.FindKeyPointsByHoffLine(lines1,w_lbhl,h_lbhl,width_closed,out1)
#        out1 = ctypes.cast(out1, ctypes.POINTER(ctypes.c_int))
#        out1_array = np.ctypeslib.as_array(out1, shape = (4,1))
#        X_Left_Method_2 = out1_array[0]
#        Y_Left_Method_2 = out1_array[1]
#        X_Right_Method_2 = out1_array[2]
#        Y_Right_Method_2 = out1_array[3]
#    else:
#        X_Left_Method_2 = 0
#        Y_Left_Method_2 = 0
#        X_Right_Method_2 = 0
#        Y_Right_Method_2 = 0
        
    #==============================================
    
    #=========== 方法　４　＝＝＝先将图像切成两部分，再分别找直线的端点坐标＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

    X_Left_Method_3, Y_Left_Method_3, X_Right_Method_3, Y_Right_Method_3 = FindKeyPointsByHoffLine2(lib_FYF,cut_img)

    #==========================＝＝求出最相邻的点，排除掉超过阈值范围的点＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝

    X_Left = np.array([X_Left_Method_0, X_Left_Method_1, X_Left_Method_2, X_Left_Method_3])
    Y_Left = np.array([Y_Left_Method_0, Y_Left_Method_1, Y_Left_Method_2, Y_Left_Method_3])
    X_Right = np.array([X_Right_Method_0, X_Right_Method_1, X_Right_Method_2, X_Right_Method_3])
    Y_Right = np.array([Y_Right_Method_0, Y_Right_Method_1, Y_Right_Method_2, Y_Right_Method_3])

    Mean_X_Left, Mean_Y_Left, Mean_X_Right, Mean_Y_Right = AbandonPointsByDistance(X_Left, Y_Left, X_Right, Y_Right)

    #================ 8月29日之前的算法＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    #filt = cut_img
    #x_Key_left, y_Key_left, x_Key_right, y_Key_right = FindWeldingKeyPoint(filt)

    #Center_Points_Key = [(x_Key_left + x_Key_right)/2 + min_x, (y_Key_left + y_Key_right)/2 + min_y]

    #x_Points_Key = [x_Key_left + min_x, x_Key_right + min_x, (x_Key_left + x_Key_right)/2 + min_x]
    #y_Points_Key = [y_Key_left + min_y, y_Key_right + min_y, (y_Key_left + y_Key_right) / 2 + min_y]

    #First_Point = (int(x_Key_left + min_x), int(y_Key_left + min_y))
    #Second_Point = (int(x_Key_right + min_x), int(y_Key_right + min_y))

    #=================9月６日更改＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    Center_Points_Key = [(Mean_X_Left + Mean_X_Right)/2 + min_x, (Mean_Y_Left + Mean_Y_Right)/2 + min_y]

    #x_Points_Key = [Mean_X_Left + min_x, Mean_X_Right + min_x, (Mean_X_Left + Mean_X_Right)/2 + min_x]
    #y_Points_Key = [Mean_Y_Left + min_y, Mean_Y_Right + min_y, (Mean_Y_Left + Mean_Y_Right) / 2 + min_y]

    #============= 转换成　float ==============================================
    First_Point = (int(Mean_X_Left + min_x), int(Mean_Y_Left + min_y))
    Second_Point = (int(Mean_X_Right + min_x), int(Mean_Y_Right + min_y))
    #First_Point = (Mean_X_Left + min_x, Mean_Y_Left + min_y)
    #Second_Point = (Mean_X_Right + min_x, Mean_Y_Right + min_y)

    points_list = [First_Point, Second_Point]
    point_size = 5
    point_color = (255, 255, 255)
    thickness = 10
    for point in points_list:
        Img_Result = cv2.circle(img, point, point_size, point_color, thickness)
    img_Adjusted = Img_Result

    # 画十字线，并显示出结果----------------------------------------------------------------
    size = 10 # 十字线大小
    color = (255, 255, 255)
    thickness = 2

     # 计算出两个拐点的中点坐标。-----------------------------------------------------
     # =============   改成float类型　＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    #Center_Point_X = int((Mean_X_Left + Mean_X_Right)/2 + min_x)
    #Center_Point_Y = int((Mean_Y_Left + Mean_Y_Right) / 2 + min_y)
    Center_Point_X = (Mean_X_Left + Mean_X_Right)/2 + min_x
    Center_Point_Y = (Mean_Y_Left + Mean_Y_Right)/2 + min_y

    #拼接字符串
    Center_Point_X_String = str(Center_Point_X)
    Center_Point_Y_String = str(Center_Point_Y)
    Str_list_1 = ['X : ', Center_Point_X_String, '       ', 'Y : ', Center_Point_Y_String]
    S_1 = ''
    S_1 = S_1.join(Str_list_1)

    # 显示出ＲＯＩ区域中的焊缝拐点的坐标位置。＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
    cv2.putText(Img_Result, 'Bo Tsing Tech', (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(Img_Result, S_1, (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    DrawCross(img_Adjusted, Center_Point_X, Center_Point_Y, size, color, thickness)
    cv2.imshow('Result', img_Adjusted)
    cv2.waitKey(0)
    
    # 计算中点到ＲＯＩ左右两边的距离
    Distance_Center_ROI_left = (Mean_X_Left + Mean_X_Right) / 2
    Distance_Center_ROI_right = width - (Mean_X_Left + Mean_X_Right) / 2
    Distance_Center_ROI_up = height - (Mean_Y_Left + Mean_Y_Right) / 2
    Distance_Center_ROI_down = (Mean_Y_Left + Mean_Y_Right) / 2

    # 计算中点到　激光线的最高点的位置＝＝＝＝
    #Distance_Center_Heightest_Line = abs(Y_0 - (Mean_Y_Left + Mean_Y_Right) / 2)

    

    # 为了滤出掉不可用的数据，准备空的数据数组。First_Points_Available = []
    global First_Points_Available_X
    global First_Points_Availabel_Y
    global Second_Points_Available_X
    global Second_Points_Available_Y

    First_Points_Available_X = []
    First_Points_Availabel_Y = []
      
    Second_Points_Available_X = []
    Second_Points_Available_Y = []

    #============20190910===添加内容＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
    #增加计算,两侧拐点的角度
    #if (Mean_X_Right != Mean_X_Left):
    #            theta_Two_Points = np.arctan((Mean_Y_Right - Mean_Y_Left) / (Mean_X_Right - Mean_X_Left))
    #else:
    #            theta_Two_Points = 0
    #theta_apro_Two_Points = np.around(theta_Two_Points, 1)

    # 增加计算,两侧拐点的距离
    Distance_Two_Points = math.sqrt((Mean_X_Left - Mean_X_Right)** 2 + (Mean_Y_Left  - Mean_Y_Right)** 2)

    #while(vid.isOpened()):
    while True:   

        #计算循环消耗时间-----------------------------------------------------
        start_1 = time.time()

        frame = fp.read()
        if frame is None:
            print("读取帧出错或文件到达末尾。")
            break 

        #ret, frame = vid.read()
        #end_1 = time.time()
        #print("每一帧读取的时间:%.4f秒"%(end_1-start_1))

        #然后对彩色图像进行灰度变换-------------------------------------------------
        #start_2 = time.time()
        Image_Gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        img = Image_Gray
        #end_2 = time.time()
        #print("彩色图像转换成灰度图的时间:%.4f秒"%(end_2-start_2))
        #(h, w) = img.shape[:2]
        #(cx, cy) = (2048//2, 1536//2) 
        #M = cv2.getRotationMatrix2D(center = (cx, cy), angle = 180, scale = 1.0)
        #img = cv2.warpAffine(img, M, (w, h))

        # 提升图像对比度First_Point_X, First_Point_Y, First_Points_Available_X , First_Points_Availabel_Y= normalizeCenter(First_Points_Available_X, First_Points_Availabel_Y, First_Point_X, First_Point_Y, queue_length, thres_drop, thres_normal)
        #First_Point_Y, First_Points_Availabel_Y = normalizeCenter(First_Points_Availabel_Y, First_Point_Y, queue_length, thres_drop, thres_normal)

        #Second_Point_X, Second_Point_Y, Second_Points_Available_X , Second_Points_Value = normalizeCenter(Second_Points_Available_X, Second_Points_Available_Y, Second_Point_X, Second_Point_Y, queue_length, thres_drop, thres_normal)
        #Second_Point_Y, Second_Points_Available_Y = normalizeCenter(Second_PointsValue = 15oint_Y, queue_length, thres_drop, thres_normal)
        
        #img = Adjust_Image(Image_Gray, Value)
        
        #==========================================================================

        # 通过前一帧图像找到的中点坐标值，重新选取ＲＯＩ区域
        # ========================== 8月３０日的算法＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        cut_img_while = img[int(Center_Points_Key[1] - Distance_Center_ROI_down):int(Center_Points_Key[1] + Distance_Center_ROI_up),
                                          int(Center_Points_Key[0] - Distance_Center_ROI_left): int(Center_Points_Key[0] + Distance_Center_ROI_right)]



        #cut_img_while = img[int(Center_Points_Key[1] - Distance_Center_ROI_down):int(Center_Points_Key[1] + Distance_Center_Heightest_Line), int(Center_Points_Key[0] - Distance_Center_ROI_left): int(Center_Points_Key[0] + Distance_Center_ROI_right)]

        #=========================================================================
        img_temp = cut_img_while.copy()
        (h, w) = img_temp.shape[:2]
        src = np.ctypeslib.as_ctypes(img_temp)
        dst = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_uint8) * w * h)
        lib_FYF.enhance_img_contrast(src, w, h, Value, dst)
        dst = ctypes.cast(dst, ctypes.POINTER(ctypes.c_uint8))
        cut_img_while = np.ctypeslib.as_array(dst, shape=img_temp.shape)
        
        #显示出找到的　中心线的位置图像。
        #cv2.namedWindow('cut_img_while', cv2.WINDOW_NORMAL)
        #cv2.imshow('cut_img_while', cut_img_while)
        #cv2.waitKey(0)
                                          
        #x_Key_left, y_Key_left, x_Key_right, y_Key_right = FindWeldingKeyPoint(cut_img_while)

        #==============方法１========通过激光中心线的位置来求取距离该曲线端点距离最远点的位置===============================================
        #=====================利用韩东的函数来求激光中心线的位置==========================================
        filt = cut_img_while
        mean = filt.mean()
        #black_limit = (int)(mean * 8)
        black_limit = (int)(mean * 2)
        if black_limit > 245:
            black_limit = 245
        if black_limit < 3:
            black_limit = 3
        
        coreline = getLineImage(lib_HD, filt, black_limit=black_limit, correct_angle=True)
        gaps = fillLineGaps(lib_HD, coreline, start_pixel=5)
        result = gaps + coreline

        # 将韩东的 result　结果转换成　激光线上各点的坐标值＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        #X_0, Y_0, X_Left_Method_0, Y_Left_Method_0, X_Right_Method_0, Y_Right_Method_0 = FindKeyPointsWithCoreLineHanDong(result)

        # 将韩东的 result　结果转换成　激光线上各点的坐标值＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        #  X_Left_Method_0, Y_Left_Method_0, X_Right_Method_0, Y_Right_Method_0 = FindKeyPointsWithCoreLineHanDong(result)
         #=================================================================================================================
        # c code
        (h, w) = result.shape[:2];
        src = np.ctypeslib.as_ctypes(result)
        out = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_int)*4)
        lib_FYF.FindKeyPointsWithCoreLine1(src,w,h,out)
        out = ctypes.cast(out, ctypes.POINTER(ctypes.c_int)) 
        out_array = np.ctypeslib.as_array(out, shape = (4,1))
        X_Left_Method_0 = out_array[0]
        Y_Left_Method_0 = out_array[1]
        X_Right_Method_0 = out_array[2]
        Y_Right_Method_0 = out_array[3]
        #=================================================================================================================
        
        #==============方法2=========利用我以前的提取中心线的函数===================================================
        X_Left_Method_1, Y_Left_Method_1, X_Right_Method_1, Y_Right_Method_1 = FindKeyPointsWithCoreLineTIANWEI(lib_FYF,cut_img_while)
#        X_Left_Method_1, Y_Left_Method_1, X_Right_Method_1, Y_Right_Method_1 = FindWeldingKeyPoint(cut_img_while)
        # 
        #==============方法3=========== 利用哈弗变换求两侧的直线端的端点＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        X_Left_Method_2, Y_Left_Method_2, X_Right_Method_2, Y_Right_Method_2 = FindKeyPointsByHoffLine1(cut_img_while)  
        
        
#        LinesByHoffLine, width_closed = GetLineByHoffLine(cut_img_while)        
#        X_Left_Method_2, Y_Left_Method_2, X_Right_Method_2, Y_Right_Method_2 = FindKeyPointsByHoffLine(LinesByHoffLine, width_closed)

        #X_Left_Method_2, Y_Left_Method_2, X_Right_Method_2, Y_Right_Method_2 = FindKeyPointsByHoffLine(cut_img_while)
      
        #==============================================
        #c code
#        if len(LinesByHoffLine) != 0:
#            (h_lbhl,w_lbhl) = LinesByHoffLine.shape[:2]
#            lines1 = np.ctypeslib.as_ctypes(LinesByHoffLine);
#            out1 = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_int)*4)
#            lib_FYF.FindKeyPointsByHoffLine(lines1,w_lbhl,h_lbhl,width_closed,out1)
#            out1 = ctypes.cast(out1, ctypes.POINTER(ctypes.c_int))
#            out1_array = np.ctypeslib.as_array(out1, shape = (4,1))
#            X_Left_Method_2 = out1_array[0]
#            Y_Left_Method_2 = out1_array[1]
#            X_Right_Method_2 = out1_array[2]
#            Y_Right_Method_2 = out1_array[3]
#        else:
#            X_Left_Method_2 = 0
#            Y_Left_Method_2 = 0
#            X_Right_Method_2 = 0
#            Y_Right_Method_2 = 0
#            
        #==============================================
        
        #=========== 方法　４　＝＝＝先将图像切成两部分，再分别找直线的端点坐标＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        #Mean_left, Mean_right, X_Left_Method_3, Y_Left_Method_3, X_Right_Method_3, Y_Right_Method_3 = FindeKeyPointsByHoffLine(cut_img_while)
        X_Left_Method_3, Y_Left_Method_3, X_Right_Method_3, Y_Right_Method_3 = FindKeyPointsByHoffLine2(lib_FYF,cut_img_while)
        
        #==========================＝＝求出最相邻的点，排除掉超过阈值范围的点＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        # 
        X_Left = np.array([X_Left_Method_0, X_Left_Method_1, X_Left_Method_2, X_Left_Method_3])
        Y_Left = np.array([Y_Left_Method_0, Y_Left_Method_1, Y_Left_Method_2, Y_Left_Method_3])
        X_Right = np.array([X_Right_Method_0, X_Right_Method_1, X_Right_Method_2, X_Right_Method_3])
        Y_Right = np.array([Y_Right_Method_0, Y_Right_Method_1, Y_Right_Method_2, Y_Right_Method_3])

        Mean_X_Left, Mean_Y_Left, Mean_X_Right, Mean_Y_Right = AbandonPointsByDistance(X_Left, Y_Left, X_Right, Y_Right)
        
#        x = [Mean_X_Left, Mean_X_Right]
#        y = [Mean_Y_Left, Mean_Y_Right]



        #============= 转换成　float ==============================================
        if (not(np.isnan(Mean_X_Left))) and (not(np.isnan(Mean_Y_Left))) and (not(np.isnan(Mean_X_Right))) and (not(np.isnan(Mean_Y_Right))) and (not (math.isinf(Mean_X_Left))) and (not (math.isinf(Mean_Y_Left))) and (not (math.isinf(Mean_X_Right))) and (not (math.isinf(Mean_X_Right))):
            #First_Point_X = int(Mean_X_Left + Center_Points_Key[0] - Distance_Center_ROI_left)
            #First_Point_Y = int(Mean_Y_Left + Center_Points_Key[1] - Distance_Center_ROI_down)
            #Second_Point_X = int(Mean_X_Right + Center_Points_Key[0] - Distance_Center_ROI_left)
            #Second_Point_Y = int(Mean_Y_Right + Center_Points_Key[1] - Distance_Center_ROI_down)
            First_Point_X = Mean_X_Left + Center_Points_Key[0] - Distance_Center_ROI_left
            First_Point_Y = Mean_Y_Left + Center_Points_Key[1] - Distance_Center_ROI_down
            Second_Point_X = Mean_X_Right + Center_Points_Key[0] - Distance_Center_ROI_left
            Second_Point_Y = Mean_Y_Right + Center_Points_Key[1] - Distance_Center_ROI_down
        
        else:
            #First_Point_X = int(get_average(First_Points_Available_X))
            #First_Point_Y = int(get_average(First_Points_Availabel_Y))
            #Second_Point_X = int(get_average(Second_Points_Available_X))
            #Second_Point_Y = int(get_average(Second_Points_Available_Y))
            First_Point_X = get_average(First_Points_Available_X)
            First_Point_Y = get_average(First_Points_Availabel_Y)
            Second_Point_X = get_average(Second_Points_Available_X)
            Second_Point_Y = get_average(Second_Points_Available_Y)


        

        # 平滑输出结果，排除掉不可用的数据。------------------------------------------------------
        # 为了滤出掉不可用的数据，准备空的数据数组　First_Points_Available。
        #First_Point_X = First_Point[0]
        #First_Point_Y = First_Point[1]

        #Second_Point_X = Second_Point[0]
        #Second_Point_Y = Second_Point[1]

        queue_length = 10
        #超过这个值就丢弃
        thres_drop =45
        # 超过这个值就和前面的值做平均 normalizeCenter(queue_array_x, queue_array_y, point_x, point_y, queue_length=10, thres_drop=60, thres_normal=25):
        thres_normal = 2

        
        # 把　First_Points_Available_X , First_Points_Availabel_Y, Second_Points_Available_X , Second_Points_Availabel_Y 转换成数组
        #First_Points_Available_X = np.array(First_Points_Available_X)
        #First_Points_Availabel_Y = np.array(First_Points_Availabel_Y)
        #Second_Points_Available_X = np.array(Second_Points_Available_X)
        #Second_Points_Availabel_Y = np.array(Second_Points_Available_X)

        First_Point_X, First_Point_Y, First_Points_Available_X , First_Points_Availabel_Y= normalizeCenter(First_Points_Available_X, First_Points_Availabel_Y, First_Point_X, First_Point_Y, queue_length, thres_drop, thres_normal)
        #First_Point_Y, First_Points_Availabel_Y = normalizeCenter(First_Points_Availabel_Y, First_Point_Y, queue_length, thres_drop, thres_normal)

        Second_Point_X, Second_Point_Y, Second_Points_Available_X , Second_Points_Availabel_Y = normalizeCenter(Second_Points_Available_X, Second_Points_Available_Y, Second_Point_X, Second_Point_Y, queue_length, thres_drop, thres_normal)
        #Second_Point_Y, Second_Points_Available_Y = normalizeCenter(Second_PointsValue = 15oint_Y, queue_length, thres_drop, thres_normal)

        # =============20190910 增加判断机制＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝


        # 显示出关键点的位置。------------------------------------------------------------
        img_Adjusted = img
        First_Point = (int(First_Point_X), int(First_Point_Y))
        Second_Point = (int(Second_Point_X), int(Second_Point_Y))
        
        points_list = [First_Point, Second_Point]
        point_size = 15
        point_color = (255, 255, 255)
        thickness = 10
        for point in points_list:
            Img_Result = cv2.circle(img_Adjusted, point, point_size, point_color, thickness)
        img_Adjusted = Img_Result
        
        # 计算出两个拐点的中点坐标。-----------------------------------------------------
        # =============   改成float类型　＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        Center_Point_X = (First_Point_X + Second_Point_X)/2
        Center_Point_Y = (First_Point_Y + Second_Point_Y) / 2
        
        # 画十字线，并显示出结果----------------------------------------------------------------
#        start_7 = time.time() 
        size = 30 # 十字线大小
        color = (255, 255, 255)
        thickness = 2
        
        #拼接字符串
        Center_Point_X_String = str(Center_Point_X)
        Center_Point_Y_String = str(Center_Point_Y)
        Str_list_1 = ['X : ', Center_Point_X_String, '       ', 'Y : ', Center_Point_Y_String]
        S_1 = ''
        S_1 = S_1.join(Str_list_1)

        #VideoName
        
        #拼接字符串
        VideoName_String = str(VideoName)
        Str_list_2 = ['Video Name : ', VideoName_String]
        S_2 = ''
        S_2 = S_2.join(Str_list_2)


        #拼接字符串
        thres_drop_String = str(thres_drop)
        thres_normal_String = str(thres_normal)
        Str_list_4 = ['thres_drop : ', thres_drop_String, '       ', 'thres_normal : ', thres_normal_String]
        S_4 = ''
        S_4 = S_4.join(Str_list_4)

        #拼接字符串
        Width_ROI_String = str(width)
        Height_ROI_String = str(height)
        Str_list_3 = ['Width of ROI Fine : ', Width_ROI_String, '       ', 'Height of ROI Fine : ', Height_ROI_String]
        S_3 = ''
        S_3 = S_3.join(Str_list_3)

        #拼接字符串
        value_String = str(Value)
        Str_list_5 = ['global variable : Value ', value_String]
        S_5 = ''
        S_5 = S_5.join(Str_list_5)

        #================ 画出图形　＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝＝
        Draw_Center_Point_X = int((First_Point_X + Second_Point_X)/2)
        Draw_Center_Point_Y = int((First_Point_Y + Second_Point_Y) / 2)
            
        cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
        #cv2.rectangle(img_Adjusted, (min_x_left, min_y_left), (max_x_left, max_y_left), (255, 255, 255), 2)
        #cv2.rectangle(img_Adjusted, (min_x_right, min_y_right), (max_x_right , max_y_right), (255, 255, 255), 2)
        cv2.putText(Img_Result, 'Bo Tsing Tech', (300, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(Img_Result, S_1, (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(Img_Result, S_2, (100, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(Img_Result, S_3, (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(Img_Result, S_4, (100, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(Img_Result, S_5, (100, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        DrawCross(img_Adjusted, Draw_Center_Point_X, Draw_Center_Point_Y, size, color, thickness)
        cv2.imshow('Result', img_Adjusted)
        cv2.waitKey(1)
        
        #Distance_Pixel = abs(First_Point_X+ min_x_right - min_x_left - Second_Point_X)
        #Real_distance = 26
        #K = Distance_Pixel/Real_distance

        #Output_result = int((Center_Point_X)/K + 30)

        # 与串口进行通讯
        #AS_device = AS.arduino_serial('/dev/ttyUSB0')
        #ret = AS_device.openPort()

        #AS_device.writePort('E70145'+bin2Hex(Output_result)+'005A0008004EFE')
        #AS_device.writePort('E7'+bin2Hex(Output_result)+'0145005A0008004EFE')

        #print("Output_result:", Output_result)
        #print("Distance Pixel:", Distance_Pixel)
        #print("K:", K)

#        Center_Points_Key[0] = int((First_Point_X + Second_Point_X)/2)
#        Center_Points_Key[1] = int((First_Point_Y + Second_Point_Y)/2)
        
        Center_Points_Key[0] = (First_Point_X + Second_Point_X)/2
        Center_Points_Key[1] = (First_Point_Y + Second_Point_Y)/2

        #Point_1_X = First_Point_X
        #Point_1_Y = First_Point_Y
        #Point_2_X = Second_Point_X
        #Point_2_Y = Second_Point_Y
        end_1 = time.time()
   
        print("------------------循环总体运行时间---------------------:%.2f秒"%(end_1-start_1))

    # Release everything if job is finished
    #vid.release()
    
    # 程序结束的时候停止图像读取线程。
    fp.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

