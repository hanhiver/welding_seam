import math
import numpy as np 
import cv2 
import ctypes
import time
import argparse

# For Arduino Serial Communication. 
import arduino_serial as AS

TEST_IMAGE = ('ssmall.png', 'sbig.png', 'rsmall.png')
#TEST_IMAGE = ('rsmall.png', )

WRITE_RESULT = False
RESIZE = 20
SLOPE_TH = 0.15

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
得到焊接表面到底面深度
"""
def getSurfaceLevel(image, max_angle = 5, min_length = 200, max_line_gap = 25):
    np.set_printoptions(precision=3, suppress=True)

    # Get the dimention of the image and then determine the certer. 
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    ret_level_left = cX
    ret_level_right = cX
    ret_bevel_top_left = -1
    ret_bevel_top_right = -1

    lines = getLines(image, min_length = min_length, max_line_gap = max_line_gap)

    zero_slope_lines_left = []
    zero_slope_lines_right = []
    max_radian = max_angle * np.pi / 180

    print('No LINE: ', type(lines))

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
            print(zero_slope_lines_left[::-1][:x])
            
            ret_level_left = np.median(zero_slope_lines_left[::-1][:x][..., 1])
            print('Level LEFT: ', ret_level_left)

            ret_bevel_top_left = np.mean(zero_slope_lines_left[::-1][:x][..., 2])
            print('Bevel Top LEFT: ', ret_bevel_top_left)


        if zero_slope_lines_right:
            zero_slope_lines_right = np.array(zero_slope_lines_right)

            # Sort the lines with length. 
            index = np.argsort(zero_slope_lines_right, axis = 0)
            index_length = index[..., 4]
            zero_slope_lines_right = zero_slope_lines_right[index_length]

            # Get the longest X lines:
            x = zero_slope_lines_right.shape[0] // 2 + 1
            print(zero_slope_lines_right[::-1][:x])
            
            ret_level_right = np.median(zero_slope_lines_right[::-1][:x][..., 1])
            print('Level RIGHT: ', ret_level_right)

            ret_bevel_top_right = np.mean(zero_slope_lines_right[::-1][:x][..., 0])
            print('Bevel Top RIGHT: ', ret_bevel_top_right)

    else:
        print('Failed to found enough surface lines. ')

    return (ret_level_left, ret_level_right, ret_bevel_top_left, ret_bevel_top_right) 


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
correct_angle: 是否对图像做矫正处理。 
"""
def getLineImage(lib, image, black_limit = 0, correct_angle = True):
    (h, w) = image.shape[:2]
    
    if correct_angle:
        kernel = np.ones((5,5),np.uint8)

        angle = getSurfaceAdjustAngle(image, min_length = 200//RESIZE)

        print('Rotate angle: ', angle)

        print('Before rotation: ', image.shape)
        image = imgRotate(image, angle)
        print('After rotation: ', image.shape)

        level = getSurfaceLevel(image, min_length = 200//RESIZE)[:2]
        print('Surface Level: ', level)

    level = (h//2, h//2)

    start = time.time()
    coreImage = getCoreImage(lib, image, black_limit = black_limit)
    lineImage = followCoreLine(lib, coreImage, level, min_gap = 100//RESIZE, black_limit = black_limit)
    end = time.time()
    print("TIME COST: ", end - start, ' seconds')

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

    cv2.line(image, (x1, y1), (x2, y2), (255, 255, 0), 3)


"""
输入图像文件，处理图像文件。
"""
def wsImagePhase(files, output = None, local_view = True):

    print('FILES: ', files)
    print("=== Start the WS Image Detecting ===")

    print("=== Read test image ===")

    display = []
    display = np.array(display)
    kernel = np.ones((3,3),np.uint8)

    print('Load C lib. ')
    so_file = './libws_c.so'
    lib = ctypes.cdll.LoadLibrary(so_file)

    lib.testlib()

    for file in files:
        print('Open file: {}'.format(file))
        frame = cv2.imread(file, cv2.IMREAD_COLOR)
        color = frame.copy()

        if type(frame) == type(None):
            print('Open file {} failed.'.format(file))
            continue

        (h, w) = frame.shape[:2]

        image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        result = getLineImage(lib, image, correct_angle = False)
        gaps = fillLineGaps(lib, result, start_pixel = 5)
        result = result + gaps 

        b_center, b_level = getBottomCenter(lib, result, bottom_thick = 30)

        np.set_printoptions(precision=10, suppress=True)
        print('Lowest point: ', b_center)
        
        frame = frame // 3 * 2

        mix_image = fill2ColorImage(lib, frame, result)
        mix_image = fill2ColorImage(lib, mix_image, gaps, fill_color = (0, 255, 0))
        drawTag(mix_image, b_center, b_level)
       
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
        gaps = cv2.cvtColor(gaps, cv2.COLOR_GRAY2RGB)

        images = np.hstack([color, mix_image, gaps])

        if display.size == 0:
            display = images.copy()
        else:
            display = np.vstack([display, images])

    if output:
        cv2.imwrite(output, display)
        print('Result file: {} saved. '.format(output))

    if local_view:
        cv2.namedWindow('Image', flags = cv2.WINDOW_NORMAL)
        #cv2.resizeWindow('Image', 1800, 1000)
        cv2.imshow('Image', display)
        k = cv2.waitKey(0)
    
    cv2.destroyAllWindows()

"""
输入视频文件，处理视频文件。
input: 输入的视频文件名称。
	   如果是数字，则是打开第n+1号系统摄像头。

output: 输出存储的视频文件名称。
"""
def wsVideoPhase(input, output, local_view = True, arduino = False):
    W = 300
    H = 250
    RESOLUTION = (W*2, H*2)
    HALF_RESOLUTION = (W, H)
    
    vid = cv2.VideoCapture(input[0])

    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video: {}".format(input))

    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Initialize the arduino serial communication. 
    if arduino is True:
        AS_device = AS.arduino_serial('/dev/ttyUSB0')
        ret = AS_device.openPort()
        if ret is False:
            print('Failed to open the Arduino Serial for communication. ')
            return

    isOutput = True if output != "" else False
    if isOutput:
        video_FourCC = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output, video_FourCC, 10, RESOLUTION)

    print("=== Start the WS detecting ===")

    print('Load C lib. ')
    so_file = './libws_c.so'
    lib = ctypes.cdll.LoadLibrary(so_file)

    lib.testlib()

    kernel = np.ones((5,5),np.uint8)

    if local_view:
        #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.namedWindow("result")
        cv2.resizeWindow("result", 800, 400)
        #cv2.moveWindow("result", 100, 100)

    while True:
        return_value, frame = vid.read()

        if type(frame) != type(None):
            
            # 根据摄像头摆放位置确定是否需要旋转图像。
            # 目前的处理逻辑是处理凸字形的焊缝折线。
            frame = np.rot90(frame, k = 0)

            # 根据摄像头摆放位置切除多余的干扰画面。
            # 目前这个设置是基于7块样板的图像进行设置。
            # 未来这里会在GUI界面中可以设置，排除不必要的干扰区域。
            (h, w) = frame.shape[:2]
            #frame = frame[0:h, w//5:w*4//5]

            if len(frame.shape) > 2:
                color_input = True
            else:
                color_input = False

            frame = cv2.resize(frame, RESOLUTION, interpolation = cv2.INTER_LINEAR)
            (h, w) = frame.shape[:2]

            #frame = cv2.medianBlur(frame, 5)

            if color_input:
                # Get the blue image. 
                b, r, g = cv2.split(frame)
                #n = 50
                #filt = (g.clip(n, n+1) - n) * 255 
                filt = g
            else:
                filt = frame

            mean = filt.mean()
            #black_limit = (int)(mean * 8)
            black_limit = (int)(mean * 5)
            if black_limit > 245:
                black_limit = 245

            if black_limit < 5:
                black_limit = 5

            print('MEAN: ', filt.mean(), ' BLACK_LIMIT: ', black_limit)

            coreline = getLineImage(lib, filt, black_limit = black_limit, correct_angle = False)
            gaps = fillLineGaps(lib, coreline, start_pixel = 5)

            result = gaps + coreline
           
            b_center, b_level = getBottomCenter(lib, result, bottom_thick = 50, noisy_pixels = 10)

            # 如果我们开启了arduino serial通讯，这里拼凑坐标传送给机器人。
            if arduino is True:

                # 这个地方的b_center就是焊缝识别得到的中点坐标。
                # 我们在这里用传统380线分辨率的做一个规一划处理。
                real_center = b_center * 380 // w
                ####################################
                # 这个地方请修改代码，是否应该将real_center的值转化为16进制的数字，
                # 通信的高地位分别怎么设置，我这里就只是用了你示例代码中的通信标志。
                ####################################
                AS_device.writePort('E7E701450124005A0008004EFE')

            if not color_input:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            mix_image = fill2ColorImage(lib, frame//2, result, fill_color = (255, 0, 0))
            mix_image = fill2ColorImage(lib, mix_image, gaps, fill_color = (0, 255, 0))

            drawTag(mix_image, b_center, b_level)

            result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
            
            drawTag(result, b_center, b_level)
            drawTag(frame, b_center, b_level)

            fill_black = np.zeros(shape = (RESOLUTION[1], RESOLUTION[0], 3))

            frame = cv2.resize(frame, HALF_RESOLUTION, interpolation = cv2.INTER_LINEAR)
            result = cv2.resize(result, HALF_RESOLUTION, interpolation = cv2.INTER_LINEAR)

            image2 = np.hstack([frame, result])
            images = np.vstack([mix_image, image2])
            
            if local_view:
                cv2.imshow("result", images)
                if cv2.waitKey(75) & 0xFF == ord('q'):
                    return False
             
            if isOutput:
                out.write(images)
        else:
            break
                
    if local_view:
        cv2.destroyAllWindows()


def main():

    parser = argparse.ArgumentParser()

    # 输入文件。
    parser.add_argument('-i', '--image', default = False, action="store_true",
                        help = '[Optional] Input video. DEFAULT: test.mp4 ')

    # 输出文件。
    parser.add_argument('-o', '--output', type = str, default = '', 
                        help = '[Optional] Output video. ')

    # 是否连接Arduino_serial进行通讯。
    parser.add_argument('-a', '--arduino', default = False, action = "store_true", 
                        help = '[Enable the Arduino Serial Communication. ')
    
    # 日志级别
    parser.add_argument('-l', '--loglevel', type = str, default = 'warning',
                        help = '[Optional] Log level. WARNING is default. ')

    # 处理单幅图像。
    parser.add_argument('-s', '--singleimages', type = str, default = None,
                        help = '[Optional] Single image files. ')

    # 是否将处理后结果显示。
    parser.add_argument('-lv', '--localview', default = False, action = "store_true",
                        help = '[Optional] If shows result to local view. ')    

    # 默认处理所有文件选项。
    parser.add_argument('input', type = str, default = None, nargs = '+',
                        help = 'Input files. ')

    FLAGS = parser.parse_args()

    if FLAGS.image:
        wsImagePhase(FLAGS.input, output = FLAGS.output)

    elif FLAGS.input:
        wsVideoPhase(input = FLAGS.input,  
                     output = FLAGS.output, 
                     local_view = FLAGS.localview,
                     arduino = FLAGS.arduino)

    else:
        print("See usage with --help.")


if __name__ == '__main__':
    main()


