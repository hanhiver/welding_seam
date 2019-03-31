import math
import numpy as np 
import cv2 
import ctypes
import time
import argparse

from matplotlib import pyplot


TEST_IMAGE = ('ssmall.png', 'sbig.png', 'rsmall.png')
#TEST_IMAGE = ('rsmall.png', )


WRITE_RESULT = False
RESIZE = 1
SLOPE_TH = 0.15

def imgRotate(image, angle):
    # Get the dimention of the image and then determine the certer. 
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Get the rotation matrix. 
    M = cv2.getRotationMatrix2D(center = (cX, cY), angle = angle, scale = 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Caculate the new bounding dimentions of the image. 
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation. 
    M[0, 2] += (new_w / 2) - cX
    M[1, 2] += (new_h / 2) - cY 

    # Perform the actrual rotation and return the image. 
    res = cv2.warpAffine(image, M, (new_w, new_h))

    return res

def imgRotate2(image, angle):
     # Get the dimention of the image and then determine the certer. 
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # Get the rotation matrix. 
    M = cv2.getRotationMatrix2D(center = (cX, cY), angle = angle, scale = 1.0)

    # Perform the actrual rotation and return the image. 
    res = cv2.warpAffine(image, M, (w, h))

    return res


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


def getSurfaceAdjustAngle(image, max_angle = 10, min_length = 200, max_line_gap = 25):
    np.set_printoptions(precision=3, suppress=True)

    # Get the dimention of the image and then determine the certer. 
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    lines = getLines(image, min_length = min_length, max_line_gap = max_line_gap)

    zero_slope_lines_left = []
    zero_slope_lines_right = []
    max_radian = max_angle * np.pi / 180

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

def getSurfaceLevel(image, max_angle = 1, min_length = 200, max_line_gap = 25):
    np.set_printoptions(precision=3, suppress=True)

    # Get the dimention of the image and then determine the certer. 
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    ret_level_left = cX
    ret_level_right = cX

    lines = getLines(image, min_length = min_length, max_line_gap = max_line_gap)

    zero_slope_lines_left = []
    zero_slope_lines_right = []
    max_radian = max_angle * np.pi / 180

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

    else:
        print('Failed to found enough surface lines. ')

    return (ret_level_left, ret_level_right) 


def getSurfaceLevel2(image, max_angle = 5, min_length = 200, max_line_gap = 25):
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

def getCorePoint(inputArray, begin, end):
    value = None
    balance = 0

    length = end - begin
    if length == 0: 
        return (0, value)
    elif length == 1:
        return (0, inputArray[begin])

    value = max(inputArray[begin:end])

    balance += inputArray[begin]
    balance -= inputArray[end]

    while begin < end:
        if balance >= 0:
            end -= 1
            balance -= inputArray[end]
        else:
            begin += 1
            balance += inputArray[begin]

    return (begin, value)  

def getCorePoint2(inputArray, begin, end):
    value = inputArray[begin:end].sum() // 2
    max_value = inputArray[begin:end].max()

    while begin < end: 
        value = value - inputArray[begin]
        if value > 0:
            begin += 1
            continue
        else:
            break

    return (begin, max_value)  


def getCoreImage(image, black_limit = 0):
    (h, w) = image.shape[:2]
    coreImage = np.zeros(shape = (h, w), dtype = np.uint8)

    scan_pos = 0

    for i in range(w):

        scan_pos = 0

        while scan_pos < h:
            
            if image[scan_pos][i] > black_limit:
                
                for seg_pos in range(scan_pos, h):
                    if image[seg_pos][i] <= black_limit:
                        break
                
                pos, value = getCorePoint2(image[..., i], scan_pos, seg_pos)
                print('DOT: ({}, {}), value: {}'.format(i, pos, value), end = '\r')
                coreImage[pos][i] = value

                scan_pos = seg_pos

            else:
                scan_pos += 1

    return coreImage

def getCoreImage2(lib, image, black_limit = 0):
    (h, w) = image.shape[:2]

    src = np.ctypeslib.as_ctypes(image)
    dst = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_uint8) * w * h)
    
    lib.getCoreImage(src, dst, h, w, black_limit)
    
    dst = ctypes.cast(dst, ctypes.POINTER(ctypes.c_uint8))
    coreImage = np.ctypeslib.as_array(dst, shape = image.shape)

    return coreImage

def followCoreLine(lib, image, ref_level, min_gap = 20, black_limit = 0):
    (h, w) = image.shape[:2]

    src = np.ctypeslib.as_ctypes(image)
    dst = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_uint8) * w * h)

    level_left, level_right = ref_level
    level_left = int(level_left)
    level_right = int(level_right)

    lib.followCoreLine(src, dst, h, w, level_left, level_right, min_gap, black_limit)

    dst = ctypes.cast(dst, ctypes.POINTER(ctypes.c_uint8))
    lineImage = np.ctypeslib.as_array(dst, shape = image.shape)

    return lineImage

def fill2ColorImage(lib, colorImage, grayImage):
    (h, w) = colorImage.shape[:2]
    colorShape = colorImage.shape

    color = np.ctypeslib.as_ctypes(colorImage)
    gray = np.ctypeslib.as_ctypes(grayImage)
    
    lib.fill2ColorImage(color, gray, h, w, 0)

    color = ctypes.cast(color, ctypes.POINTER(ctypes.c_uint8))
    mergedImage = np.ctypeslib.as_array(color, shape = colorShape)

    return mergedImage


def getLineImage(lib, image, correct_angle = True):
    (h, w) = image.shape[:2]
    
    #if RESIZE != 1:
    #    image = cv2.resize(image, (h//RESIZE, w//RESIZE))

    if correct_angle:
        kernel = np.ones((5,5),np.uint8)

        angle = getSurfaceAdjustAngle(image, min_length = 200//RESIZE)

        print('Rotate angle: ', angle)

        print('Before rotation: ', image.shape)
        image = imgRotate2(image, angle)
        print('After rotation: ', image.shape)

        level = getSurfaceLevel2(image, min_length = 200//RESIZE)[:2]
        print('Surface Level: ', level)

    level = (h//2, h//2)

    start = time.time()
    #coreImage = getCoreImage(image, black_limit = 0)
    coreImage = getCoreImage2(lib, image, black_limit = 0)
    lineImage = followCoreLine(lib, coreImage, level, min_gap = 100//RESIZE)
    end = time.time()
    print("TIME COST: ", end - start, ' seconds')

    return lineImage

def getBevelTopCenter(lineImage, max_angle = 5, min_length = 200, max_line_gap = 25):
    kernel = np.ones((5,5),np.uint8)

    dilate = cv2.dilate(lineImage, kernel, iterations = 1)

    bevel_top = getSurfaceLevel2(dilate, 
                                 max_angle = max_angle, 
                                 min_length = 100,
                                 max_line_gap = max_line_gap)
    bevel_top = bevel_top[2:]

    #bevel_top_center = int((bevel_top[0] + bevel_top[1]) / 2)
    #return bevel_top_center

    return int(bevel_top[0]), int(bevel_top[1])

def getBevelTop(lib, coreImage, judgeLength = 10):
    (h, w) = coreImage.shape[:2]
    #coreImageShape = coreImage.shape

    coreLine = np.ctypeslib.as_ctypes(coreImage)
    slope = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_float) * w)
    
    lib.getBevelTop(coreLine, slope, h, w)

    slope = ctypes.cast(slope, ctypes.POINTER(ctypes.c_float))
    slope_array = np.ctypeslib.as_array(slope, shape = (w,))

    return slope_array

def coreLine2Index(lib, coreImage):
    (h, w) = coreImage.shape[:2]
    
    coreLine = np.ctypeslib.as_ctypes(coreImage)
    index = ctypes.create_string_buffer(ctypes.sizeof(ctypes.c_int) * w)
    
    lib.coreLine2Index(coreLine, h, w, index)

    index = ctypes.cast(index, ctypes.POINTER(ctypes.c_int))
    index_array = np.ctypeslib.as_array(index, shape = (w,))

    return index_array

def getBevelBottom(lib, coreImage):
    index_array = coreLine2Index(lib, coreImage)

    return index_array.argmin()


def wsImagePhase(files, output = None):

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

        image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        result = getLineImage(lib, image, correct_angle = False)
        
        #slope_array = getBevelTop(lib, result)
        #slope_array = np.arctan2(slope_array)

        index_array = coreLine2Index(lib, result)
        bottom_index = np.where(index_array < (index_array.min()+10))

        center = np.mean(bottom_index)
        center = int(center)

        np.set_printoptions(precision=10, suppress=True)
        print('Lowest point: ', center)
        print(bottom_index)
        
        #pyplot.plot(index_array)
        #pyplot.show()
        

        #cv2.imwrite('line.jpg', result)

        #bevel_top = getBevelTopCenter(image)
        #bevel_top_center = (bevel_top[0] + bevel_top[1]) // 2
        #print("BEVEL: ", bevel_top_center)
        #result = cv2.dilate(result, kernel, iterations = 1)
        #color = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        frame = frame // 3 * 2

        #images = np.hstack([image, result])
        mix_image = fill2ColorImage(lib, frame, result)
        cv2.line(mix_image, (center, 200), (center, 1800), (255, 255, 0), 1)
        
        """
        cv2.line(mix_image, (bevel_top_center, 200), (bevel_top_center, 1800), (255, 255, 0), 8)
        cv2.line(mix_image, (bevel_top[0], 200), (bevel_top[0], 1800), (0, 255, 0), 3)
        cv2.line(mix_image, (bevel_top[1], 200), (bevel_top[1], 1800), (0, 255, 0), 3)
        """

        #######################################################
        #lines = getLines(frame)
        #for line in lines:
        #    x1, y1, x2, y2 = line[0]
        #    cv2.line(mix_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        #######################################################


        #result = cv2.dilate(result, kernel, iterations = 1)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

        images = np.hstack([color, mix_image, result])

        if display.size == 0:
            display = images.copy()
        else:
            display = np.vstack([display, images])

    if output:
        cv2.imwrite(output, display)
        print('Result file: {} saved. '.format(output))

    cv2.namedWindow('Image', flags = cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Image', 1800, 1000)
    cv2.imshow('Image', display)
    k = cv2.waitKey(0)
    
    cv2.destroyAllWindows()


def wsVideoPhase(input, output, local_view = True):

    vid = cv2.VideoCapture(input[0])

    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video: {}".format(input))

    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    isOutput = True if output != "" else False
    if isOutput:
        video_FourCC = cv2.VideoWriter_fourcc(*'XVID')
        #video_FourCC = -1
        #print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        #print("!!! TYPE:", output_path, video_FourCC, video_fps, video_size)
        out = cv2.VideoWriter(output, video_FourCC, 10, (1200, 800))

    print("=== Start the WS detecting ===")

    print('Load C lib. ')
    so_file = './libws_c.so'
    lib = ctypes.cdll.LoadLibrary(so_file)

    lib.testlib()

    kernel = np.ones((3,3),np.uint8)

    if local_view:
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("result", 800, 400)
        cv2.moveWindow("result", 100, 100)

    while True:
        return_value, frame = vid.read()

        if type(frame) != type(None):
            frame = cv2.resize(frame, (1200, 800), interpolation = cv2.INTER_LINEAR)
            image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            #print("COLOR: ", image)
            #result = frame
            #result = wsImagePhase(lib, image, correct_angle = False)
            result = getLineImage(lib, image, correct_angle = False)
            #image = Image.fromarray(frame)
            #image = dpool_detect_car(client, image)
            #result = np.asarray(image)
            #result = cv2.dilate(result, kernel, iterations = 1)
            #color = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

            index_array = coreLine2Index(lib, result)
            bottom_index = np.where(index_array < (index_array.min()+10))

            center = np.mean(bottom_index)
            center = int(center)

            #image = image // 2
            frame = frame // 3 * 2

            #images = np.hstack([image, result])
            images = fill2ColorImage(lib, frame, result)

            cv2.line(images, (center, 200), (center, 600), (255, 255, 0), 1)


            if local_view:
                cv2.imshow("result", images)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return False
             
            if isOutput:
                out.write(images)
        else:
            break
                
    if local_view:
        cv2.destroyAllWindows()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--image', default = False, action="store_true",
                        help = '[Optional] Input video. DEFAULT: test.mp4 ')

    parser.add_argument('-o', '--output', type = str, default = '', 
                        help = '[Optional] Output video. ')
    
    parser.add_argument('-l', '--loglevel', type = str, default = 'warning',
                        help = '[Optional] Log level. WARNING is default. ')

    parser.add_argument('-s', '--singleimages', type = str, default = None,
                        help = '[Optional] Single image files. ')

    parser.add_argument('-lv', '--localview', default = False, action = "store_true",
                        help = '[Optional] If shows result to local view. ')    

    parser.add_argument('input', type = str, default = None, nargs = '+',
                        help = 'Input files. ')

    FLAGS = parser.parse_args()

    if FLAGS.image:
        wsImagePhase(FLAGS.input, output = FLAGS.output)

    elif FLAGS.input:
        wsVideoPhase(input = FLAGS.input,  
                     output = FLAGS.output, 
                     local_view = FLAGS.localview)

    else:
        print("See usage with --help.")


if __name__ == '__main__':
    main()


