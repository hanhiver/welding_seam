import numpy as np
import cv2
import ctypes
import time
import sys
import logging

sys.path.append("..")
import wslib.clib.BQ_clib as clib 
import wslib.pylib.BQ_downwardWelding as dwelding
from wslib.pylib.loggerManager import LoggerManager

time_stamp = time.time()

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
    def __init__(self, logger_manager, 
                 bottom_thick = 100, noisy_pixels = 20, 
                 #thres_drop = 100, thres_normal = 50, move_limit = 3, 
                 so_file = "./wslib/clib/libws_c.so"
                 ):
        self.logger = logger_manager.get_logger("BQ_WsPos")
        self.lib = clib.initCLib(so_file)
        self.logger.debug("c库函数加载完成。")

        self.bottom_thick = bottom_thick
        self.noisy_pixels = noisy_pixels
        self.clahe = cv2.createCLAHE(clipLimit = 40, tileGridSize = (8, 8))
        self.logger.debug("各参数设置初始化完成。")

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
        time_stamp = time.time()

        if image is not None:
            self.image = image 

        # 图像灰度转换
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        self.logger.debug("图像色域切割完成。")

        self.clahe.apply(self.gray)
        self.logger.debug("图像对比度增强完成。")

        # 黑场检测和阈值切割
        mean = self.gray.mean()
        self.black_limit = (int)(mean * 5)
        if self.black_limit > 245:
            self.black_limit = 245
        if self.black_limit < 3:
            self.black_limit = 3

        if self.logger.getEffectiveLevel() >= logging.INFO:
            time_curr = time.time()
            time_due = (time_curr - time_stamp) * 1000
            time_stamp = time_curr
            self.logger.info("        {:3.3f} ms 图像增强操作".format(time_due))

        self.coreline = clib.getLineImage(
                                self.lib, 
                                self.gray, 
                                black_limit = self.black_limit)
        self.logger.debug("轮廓识别完成。")
        
        if self.logger.getEffectiveLevel() >= logging.INFO:
            time_curr = time.time()
            time_due = (time_curr - time_stamp) * 1000
            time_stamp = time_curr
            self.logger.info("        {:3.3f} ms 轮廓识别".format(time_due))

        self.gaps = clib.fillLineGaps(
                                self.lib, 
                                self.coreline, 
                                start_pixel = 5)
        self.logger.debug("缺损填补完成。")
        
        self.coreline = self.gaps + self.coreline

        if self.logger.getEffectiveLevel() >= logging.INFO:
            time_curr = time.time()
            time_due = (time_curr - time_stamp) * 1000
            time_stamp = time_curr
            self.logger.info("        {:3.3f} ms 缺损填补。".format(time_due))

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
        time_stamp = time.time()
        if coreline_image is not None:
            self.coreline = coreline_image

        # 从baseline获取底部中点位置，底部中点高度和包围位置。
        self.center, self.level, self.bound = dwelding.getBottomCenter(
                                                self.lib, 
                                                self.coreline, 
                                                bottom_thick = self.bottom_thick, 
                                                noisy_pixels = self.noisy_pixels)
        
        self.logger.debug("中点位置识别完成。")

        if self.logger.getEffectiveLevel() >= logging.INFO:
            time_curr = time.time()
            time_due = (time_curr - time_stamp) * 1000
            time_stamp = time_curr
            self.logger.info("        {:3.3f} ms 中点位置识别。".format(time_due))

        return self.center, self.level, self.bound 

    def phaseImage(self, image):
        self.image = image
        self.img2baseline()
        center, level, bound = self.getBottomCenter()

        return center, level, bound

    def fillCoreline2Image(self, image, X = 0, Y = 0, fill_color = (255, 0, 0)):
        time_stamp = time.time()
        
        (h, w) = image.shape[:2]
        extend_coreline = np.zeros(shape = (h,w), dtype = np.uint8)
        extend_coreline[Y:Y+self.coreline.shape[0], X:X+self.coreline.shape[1]] += self.coreline
        result = clib.fill2ColorImage(self.lib, image, extend_coreline, fill_color = fill_color)

        if self.logger.getEffectiveLevel() >= logging.INFO:
            time_curr = time.time()
            time_due = (time_curr - time_stamp) * 1000
            time_stamp = time_curr
            self.logger.info("        {:3.3f} ms 图像标记。".format(time_due))

        return result


class PosNormalizer():

    def __init__(self, logger_manager,  
                 thres_drop = 100, thres_normal = 50, move_limit = 3):
        self.logger = logger_manager.get_logger("BQ_WsPos")
        self.value_array = []
        self.dropped_array = []
        self.thres_drop = thres_drop
        self.thres_normal = thres_normal
        self.move_limit = move_limit
        self.last_pos = None
        self.logger.debug("各参数设置初始化完成。")

    def normalizeCenter(self, center):
        time_stamp = time.time()

        self.center, self.value_array, self.dropped_array = dwelding.normalizeCenter(
                                self.value_array, center, 
                                thres_drop = self.thres_drop, 
                                thres_normal = self.thres_normal,
                                move_limit = self.move_limit, 
                                dropped_array = self.dropped_array)
        self.logger.debug("位置输出平滑降噪完成。")
        #self.logger.debug("  - value_array: {}, dropped_array: {}".format(self.value_array, self.dropped_array))
        
        if self.last_pos is None:
            diff = 0
        else:
            diff = self.center - self.last_pos
            
        self.last_pos = self.center

        if self.logger.getEffectiveLevel() >= logging.INFO:
            time_curr = time.time()
            time_due = (time_curr - time_stamp) * 1000
            time_stamp = time_curr
            self.logger.info("        {:3.3f} ms 位置输出平滑降噪。".format(time_due))

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





