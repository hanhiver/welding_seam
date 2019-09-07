import numpy as np
import cv2
import ctypes
import time
import argparse
import sys

sys.path.append("..")
import wslib.clib.BQ_clib as clib 
import wslib.pylib.BQ_downwardWelding as dwelding

"""
BQ_wsPos模块
该模块完成从获取的原始图像到最终识别位置的输出。
"""
class BQ_WsPos():
    """
    初始化函数
    bottom_thick: 底部识别厚度。
    noisy_pixels: 底部中前后出现偏差的多少像素当做噪音抛弃。 
    thres_drop: 超出此偏移量将被认为是大干扰滤除。
    thres_normal: 超出此偏移量将被认为是小的干扰，会被记入队列平均，但是输出还是之前量。
    move_limit: 如果输出量相对前面的值移动范围小于此值将不会变化。
    so_file: 编译好的c库文件。
    """
    def __init__(self, 
                 bottom_thick = 100, noisy_pixels = 20, 
                 #thres_drop = 100, thres_normal = 50, move_limit = 3, 
                 so_file = "./wslib/clib/libws_c.so"
                 ):
        
        self.lib = clib.initCLib(so_file)
        self.bottom_thick = bottom_thick
        self.noisy_pixels = noisy_pixels

    """
    测试库文件是否正常加载。
    """
    def testlib(self):
        self.lib.testlib()

    """
    输入原始图像。
    输入image空的话自动使用self.image
    输出将自动生成如下图像：
    self.gray: 分色后的灰度图像
    self.coreline: 经过填补的轮廓线。
    self.gaps: 程序自动识别填补部分。
    函数自动返回self.coreline.
    """
    def img2baseline(self, image = None):
        if image is not None:
            self.image = image 

        # 图像灰度转换
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

        # 黑场检测和阈值切割
        mean = self.gray.mean()
        self.black_limit = (int)(mean * 5)
        if self.black_limit > 245:
            self.black_limit = 245
        if self.black_limit < 3:
            self.black_limit = 3

        self.coreline = clib.getLineImage(
                                self.lib, 
                                self.gray, 
                                black_limit = self.black_limit)
        
        self.gaps = clib.fillLineGaps(
                                self.lib, 
                                self.coreline, 
                                start_pixel = 5)
        
        self.coreline = self.gaps + self.coreline

        return self.coreline

    """
    根据coreline识别焊缝位置输出。
    coreline_image为空的话自动使用self.coreline
    输出 (center, level, bound)
    center: 焊缝中心位置 
    level: 焊缝中心位置高度
    bound: 焊缝识别宽度
    """
    def getBottomCenter(self, coreline_image = None):
        if coreline_image is not None:
            self.coreline = coreline_image

        # 从baseline获取底部中点位置，底部中点高度和包围位置。
        self.center, self.level, self.bound = dwelding.getBottomCenter(
                                                self.lib, 
                                                self.coreline, 
                                                bottom_thick = self.bottom_thick, 
                                                noisy_pixels = self.noisy_pixels)
        
        """
        # 将center的输出值进行normalize处理，消除尖峰噪音干扰。
        self.center, self.value_array, self.dropped_array = dwelding.normalizeCenter(
                                                self.value_array, center, 
                                                thres_drop = self.thres_drop, 
                                                thres_normal = self.thres_normal,
                                                move_limit = self.move_limit, 
                                                dropped_array = self.dropped_array)
        """
        return self.center, self.level, self.bound 

    def normalized(center = None, ):
        pass

    def phaseImage(self, image):
        self.image = image 
        self.img2baseline()
        center, level, bound = self.getBottomCenter()

        return center, level, bound

    def fillCoreline2Image(self, image, X = 0, Y = 0, fill_color = (255, 0, 0)):
        (h, w) = image.shape[:2]
        extend_coreline = np.zeros(shape = (h,w), dtype = np.uint8)
        extend_coreline[Y:Y+self.coreline.shape[0], X:X+self.coreline.shape[1]] += self.coreline
        result = clib.fill2ColorImage(self.lib, image, extend_coreline, fill_color = fill_color)

        return result


class PosNormalizer():

    def __init__(self, thres_drop = 100, thres_normal = 50, move_limit = 3):
        self.value_array = []
        self.dropped_array = []
        self.thres_drop = thres_drop
        self.thres_normal = thres_normal
        self.move_limit = move_limit
        self.last_pos = None

    def normalizeCenter(self, center):
        self.center, self.value_array, self.dropped_array = dwelding.normalizeCenter(
                                self.value_array, center, 
                                thres_drop = self.thres_drop, 
                                thres_normal = self.thres_normal,
                                move_limit = self.move_limit, 
                                dropped_array = self.dropped_array)
        
        if self.last_pos is None:
            diff = 0
        else:
            diff = self.center - self.last_pos
            
        self.last_pos = self.center

        return self.center, diff 


"""
BQ_wsPos模块测试程序。
"""
def main():
    ws = BQ_WsPos() 
    ws.testlib()
    img_raw = cv2.imread("./clib/rsmall.png", 3)
    """
    coreline = ws.img2baseline(img_raw)
    center, level, bound = ws.getBottomCenter()
    """
    center, level, bound = ws.phaseImage(img_raw)
    print(center, level, bound)

    result = ws.coreline
    cv2.namedWindow("BQ_WsPos Test", cv2.WINDOW_NORMAL)
    cv2.imshow("BQ_WsPos Test", result)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()





