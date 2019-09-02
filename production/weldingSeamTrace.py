import math
import numpy as numpy
import ctypes
import time

"""
初始化C函数动态连接库的调用
"""
def initCLib(so_file = "./libws_c.so"):
	
    print('Load C lib. ')
    so_file = './libws_c.so'
    lib = ctypes.cdll.LoadLibrary(so_file)
    
    lib.testlib()

    return lib	


"""
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

    lib.followCoreLine(src, dst, h, w, level_left, level_right, min_gap, black_limit)

    dst = ctypes.cast(dst, ctypes.POINTER(ctypes.c_uint8))
    lineImage = np.ctypeslib.as_array(dst, shape = image.shape)
    #lineImage = dst

    return lineImage

"""
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
从原始输入图像得到最终识别结果的轮廓线。
black_limit: 小于等于这个值的点会被视作黑色。
"""
def getLineImage(lib, image, black_limit = 0):
    (h, w) = image.shape[:2]
    level = (h//2, h//2)

    coreImage = getCoreImage(lib, image, black_limit = black_limit)
    lineImage = followCoreLine(lib, coreImage, level, min_gap = 100//RESIZE, black_limit = black_limit)

    return lineImage

"""
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

