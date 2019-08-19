#-*- coding: UTF-8 -*-
import numpy as np # 引入numpy 用于矩阵运算
from matplotlib import pyplot as plt
import cv2 # 引入opencv库函数
import math
import glob
import copy


leftButtonDownFlag = False
point1x,point1y,point2x,point2y = 0, 0, 0, 0
frameNum = 0
frameSaved = False
framePre = 0
initLaserPoints = list()
laserPointsPre = list()

def on_mouse(event, x, y, flags, param):
    global point1x,point1y,point2x,point2y,leftButtonDownFlag
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        leftButtonDownFlag = True
        point1x = x
        point1y = y
    elif event == cv2.EVENT_MOUSEMOVE:         #左键移动
        if(leftButtonDownFlag==True):
            point2x = x
            point2y = y
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        leftButtonDownFlag = False
        point2x = x
        point2y = y

def ChoseRoi():
    cv2.setMouseCallback('image_win', on_mouse)

def grey_scale(image):
    img_gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    rows,cols = img_gray.shape
    flat_gray = img_gray.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    print('A = %d,B = %d' %(A,B))
    output = np.uint8(255 / (B - A) * (img_gray - A) + 0.5)
    return output

# 平滑处理
def smoothing(frame, size=1):
    # 高斯滤波
    if size<=0:
        size = 1
    # smmothing_roi = cv2.medianBlur(frame,size)
    smoothing_roi = cv2.GaussianBlur(frame,(size,size),1.5)
    # 中值滤波
    # smoothing_roi = cv2.medianBlur(frame, 3)    
    return smoothing_roi

# 图像增强
def image_reinforce(img,clipLimitValue,reinforceValue):
    # 使用全局直方图均衡化
    # equa = cv2.equalizeHist(img)
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit=clipLimitValue, tileGridSize=(reinforceValue, reinforceValue))
    # 限制对比度的自适应阈值均衡化
    dst = clahe.apply(img)
    
    return dst

# 骨架提取
def skelenton(img):
    None

# 返回图像的计数器以及对应的哪些点是共线的
def HTLine(image, stepTheta=1, stepRho=1):
    # 宽、高
    rows,cols = image.shape
    #图像中可能出现的最大垂线的长度
    L = round(math.sqrt(pow(rows-1,2.0)+pow(cols-1,2.0)))+1
    #初始化投票器
    numtheta = int(180.0/stepTheta)
    numRho = int(2*L/stepRho+1)
    accumulator = np.zeros((numRho,numtheta),np.int32)
    #建立字典
    accuDict = {}
    for k1 in range(numRho):
        for k2 in range(numtheta):
            accuDict[(k1,k2)]=[]
    # 投票计数
    for y in range(rows):
        for x in range(cols):
            if(image[y][x] == 255): # 只对边缘点做霍夫变换
                # 对每个角度计算对应的Rho值
                for m in range(numtheta):
                    rho = x*math.cos(stepTheta*m/180.0*math.pi)+y*math.sin(stepTheta*m/180.0*math.pi)
                    # 计算投票哪个区域
                    n = int(round(rho+L)/stepRho)
                    # 投票+1
                    accumulator[n,m] += 1
                    # 记录这个点
                    accuDIct[(n,m)].append((y,x))
    return accumulator,accuDict
    
# 时间片二值化
def timeSlotThreshold(frameNow, threshold1, threshold2):
    global framePre
    frameNowCopy = frameNow.copy()
    rows,cols = frameNowCopy.shape
    for i in range(rows):
        for j in range(cols):
            change = abs(int(frameNowCopy[i][j])-int(framePre[i][j]))
            if(change < threshold1 and frameNowCopy[i][j] > threshold2):
                frameNowCopy[i][j] = 255
            else:
                frameNowCopy[i][j] = 0
    return frameNowCopy
    # for pngfile in glob.glob('/home/gq5251/py_code/frame*.png'):
        # print('frameNum is ',pngfile)


# 水平扫描，去除
def HorizontalScan(image):
    None

# 每隔十个像素取一个平均值，获取最大的平均值，将最大平均值之外的像素都置255
# def threshold_10(image):
#     image_copy = image.copy()
#     rows,cols = image_copy.shape
#     depth = int(rows/10)-2
#     li = {}
#     for i in range(10,cols-10):
#         for j in range(1,depth-10):
#             j = j * 10
#             li[((int(image_copy[j][i])+int(image_copy[j+1][i])+int(image_copy[j+2][i])+int(image_copy[j+3][i])+
#             int(image_copy[j+4][i])+int(image_copy[j+5][i])+int(image_copy[j+6][i])+int(image_copy[j+7][i])+
#             int(image_copy[j+8][i])+int(image_copy[j+9][i]))/10)]=[image_copy[j][i],image_copy[j+1][i],image_copy[j+2][i],image_copy[j+3][i],
#             image_copy[j+4][i],image_copy[j+5][i],image_copy[j+6][i],image_copy[j+7][i],
#             image_copy[j+8][i],image_copy[j+9][i]]
#         avg = max(li)
#         li = []
#         for j in range(10,rows-10):
#             if(image_copy[j][i] > avg):
#                 image_copy[j][i] = 255
#             else:
#                 image_copy[j][i] = 0
#     return image_copy
#计算图像灰度直方图
def calcGrayHist(image):
    #灰度图像矩阵的宽高
    rows,cols = image.shape
    #存储灰度直方图
    grayHist = np.zeros([1,256],np.uint32)
    for r in range(rows):
        for c in range(cols):
            grayHist[0][image[r][c]] +=1
    return grayHist         
def ostu(image):
    rows,cols = image.shape
    #计算图像的灰度直方图
    grayHist = calcGrayHist(image)
    #归一化灰度直方图
    uniformGrayHist = grayHist/float(rows*cols)
    #计算零阶累积矩和一阶累积矩
    zeroCumuMoment = np.zeros([1,256],np.float32)
    oneCumuMoment = np.zeros([1,256],np.float32)
    for k in range(256):
        if k == 0:
            zeroCumuMoment[0][k] = uniformGrayHist[0][0]
            oneCumuMoment[0][k] = (k+1)*uniformGrayHist[0][0]
        else:
            zeroCumuMoment[0][k] = zeroCumuMoment[0][k-1] + uniformGrayHist[0][k]
            oneCumuMoment[0][k] = oneCumuMoment[0][k-1] + k*uniformGrayHist[0][k]
    #计算类间方差  
    variance = np.zeros([1,256],np.float32)
    for k in range(255):
        if zeroCumuMoment[0][k] == 0:
            variance[0][k] = 0
        else:
            variance[0][k] = math.pow(oneCumuMoment[0][255]*zeroCumuMoment[0][k] - oneCumuMoment[0][k],2)/(zeroCumuMoment[0][k]*(1.0-zeroCumuMoment[0][k]))
    #找到阈值
    threshLoc = np.where(variance[0][0:255] == np.max(variance[0][0:255]))
    thresh = threshLoc[0]
    #阈值处理
    threshold = np.copy(image)
    threshold[threshold > thresh] = 255
    threshold[threshold <= thresh] = 0
    return threshold

def nothing(x):
    pass

def centerExtraction(image):
    rows,cols = image.shape
    li = list()
    for i in range(5, cols-5,3):
        sumOfWhite = 0
        indexOfWhite = 0
        for j in range(10,rows-10):
            if(image[j][i] == 255):
                sumOfWhite+=1
        for j in range(10,rows-10):
            if(image[j][i] == 255):
                indexOfWhite+=1
                if(indexOfWhite == int(sumOfWhite/2)):
                    li.append((i,j))
                    break
    return li

# 获取初始激光线 (红色像素)
def getInitLaserLine(image,laserWidth):
    rows,cols,channel = image.shape
    laserFilter = np.zeros((rows,cols),np.uint8)
    li = list()
    for i in range(0,cols):
        for j in range(0, rows):
            if(image[j][i][2] == 255 and image[j][i][0]==0):
                for x in range(j-laserWidth,j+laserWidth):
                    laserFilter[x][i] = 1
    return laserFilter

# 去除小面积
def RemoveSmallRegion(src, threshold):
    contours, hierarchy = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area<threshold:
            cv2.drawContours(src, [contours[i]], 0,0,-1)
    return src

def main():
    # WeldingVideo12
    # global initLaserPoints
    #cap = cv2.VideoCapture('/home/gq5251/py_code/Video/WeldingVideo003.avi')
    cap = cv2.VideoCapture('../../wsdata/2.avi')
    ret, img = cap.read()
    global frameNum,frameSaved,leftButtonDownFlag,framePre
        
    while(1):
        ret, img = cap.read()
        if(ret == True):
            # cv2.namedWindow('image_win',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
            cv2.namedWindow('image_win',cv2.WINDOW_NORMAL)
            cv2.createTrackbar('ThreshValue','image_win',100,255,nothing)
            threshValue=cv2.getTrackbarPos('ThreshValue','image_win')
            cv2.createTrackbar('sizeValue','image_win',15,20,nothing)
            sizeValue=cv2.getTrackbarPos('sizeValue','image_win')
            cv2.createTrackbar('dilateKernel','image_win',5,20,nothing)
            dilateValue=cv2.getTrackbarPos('dilateKernel','image_win')
            cv2.createTrackbar('erosionKernel','image_win',5,20,nothing)
            erosionValue=cv2.getTrackbarPos('erosionKernel','image_win')
            cv2.createTrackbar('reinforceKernel','image_win',30,50,nothing)
            reinforceValue=cv2.getTrackbarPos('reinforceKernel','image_win')
            cv2.createTrackbar('clipLimitValue','image_win',10,30,nothing)
            clipLimitValue=cv2.getTrackbarPos('clipLimitValue','image_win')
            cv2.createTrackbar('smallRegionValue','image_win',80,1000,nothing)
            smallRegionValue=cv2.getTrackbarPos('smallRegionValue','image_win')
            ChoseRoi()
            if(point1x != 0 and point2x != 0 and leftButtonDownFlag == False):
                frameNum += 1
                # if(frameNum==2):
                #     frameSaved = True
                # 获取ROI区域
                cv2.rectangle(img, (point1x,point1y), (point2x,point2y), (0,255,0), 3)
                img_roi=img[point1y:point2y,point1x:point2x]
                # 对ROI区域进行图像处理
                gray = cv2.cvtColor(img_roi, cv2.COLOR_RGB2GRAY)

                # if(frameNum>1):
                #     gray = gray*initLaserPoints
                # gray = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
                # laplacian = cv2.Laplacian(gray,cv2.CV_64F)
                if(sizeValue%2!=0):
                    smoothing_roi = smoothing(gray,sizeValue)
                else:
                    smoothing_roi = smoothing(gray,sizeValue-1)
                
                gray = image_reinforce(smoothing_roi,clipLimitValue/10,reinforceValue)
                # ret,gray = cv2.threshold(gray,threshValue,255, cv2.THRESH_BINARY)
                ret,binary = cv2.threshold(gray,threshValue,255,cv2.THRESH_OTSU)
                # 先膨胀再腐蚀
                dkernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilateValue, dilateValue))  # 矩形结构
                binary = cv2.dilate(binary, dkernel)
                ekernel = np.ones((erosionValue, erosionValue), np.uint8)
                erode = cv2.erode(binary, ekernel)
                imageWithoutSmall = RemoveSmallRegion(erode,smallRegionValue)
                points = centerExtraction(imageWithoutSmall)
                
                # 时间片二值化
                # cv2.imwrite('frame'+str(frameNum)+'.png', gray)
                # if(frameSaved == True):
                #     binary = timeSlotThreshold(gray,10,35)
                #     # ret, binary = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
                #     # binary = threshold_10(gray)
                # else:
                #     # ret, binary = cv2.threshold(gray, 255*0.2, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
                #     binary = gray
                
                # timeSlotThreshold()
                # 
                # binary = cv2.adaptiveThreshold(drawing, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,-10)
                
                
                # edge_output = cv2.Canny(binary, 100, 300, apertureSize = 5)
                # drawing = np.zeros(gray.shape[:], dtype=np.uint8)
                # 霍夫直线检测
                # accumulator,accuDict = HTLine(edge_output,1,1)
                # lines = cv2.HoughLinesP(edge_output, 1, np.pi / 180, 100, minLineLength=10,maxLineGap=10)  #函数将通过步长为1的半径和步长为π/180的角来搜索所有可能的直线
                # for i in range(0,len(lines)):
                #     rho,theta = lines[i][0][0],lines[i][0][1]
                #     a  =  np.cos(theta)
                #     # print("theta is ", a)
                #     b  =  np.sin(theta)
                #     x0  =  a*rho
                #     y0  =  b*rho
                #     x1  =  int(x0  +  1000*(-b))
                #     y1  =  int(y0  +  1000*(a))
                #     x2  =  int(x0  -  1000*(-b))
                #     y2  =  int(y0  -  1000*(a))
                #     if(a < 0.1):
                #         cv2.line(drawing,(x1,y1),(x2,y2),(255,255,255),2)
                # for line in lines:
                #     x1,y1,x2,y2 = line[0]
                #     cv2.line(edge_output,(x1,y1),(x2,y2),(0,0,255),2)

                # th = cv2.adaptiveThreshold(drawing, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,51,-10)
                # 处理后的图像覆盖到原图像
                red = (0,0,255)
                backface=cv2.cvtColor(binary,cv2.COLOR_GRAY2BGR)
                for i,p in enumerate(points):
                    # cv2.circle(backface,p,2,red)
                    if(i > 0):
                        cv2.line(backface,points[i-1],p,red,2)
                img[point1y:point2y,point1x:point2x] = backface
                # if(frameNum==1):
                #     initLaserPoints = getInitLaserLine(backface,5)
                # ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) 
                # Gaussian_roi = cv2.GaussianBlur(img_roi,(3,3),0)
                # edges = cv2.Canny(Gaussian_roi, 50, 80, apertureSize = 3)
                framePre = gray
                cv2.imwrite('frame'+str(frameNum)+'.png', backface)
            cv2.imshow('image_win', img)
            
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
        else:
            break

    # 释放VideoCapture
    cap.release()
    # 销毁所有的窗口
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
