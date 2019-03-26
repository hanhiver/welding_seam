import math
import numpy as np 
import cv2 
import ctypes
import time


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

    ret_level_left = 0
    ret_level_right = 0

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

    lib.followCoreLine(src, dst, h, w, ref_level, min_gap, black_limit)

    dst = ctypes.cast(dst, ctypes.POINTER(ctypes.c_uint8))
    lineImage = np.ctypeslib.as_array(dst, shape = image.shape)

    return lineImage


def main():
    print("=== Start the WS detecting ===")

    print('Load C lib. ')
    so_file = './libws_c.so'
    lib = ctypes.cdll.LoadLibrary(so_file)

    lib.testlib()

    print("=== Read test image ===")

    display = []
    display = np.array(display)

    for file in TEST_IMAGE:
        print('Open file: {}'.format(file))
        image_gray = cv2.imread(file, 0)
        #image_color = cv2.imread(file, cv2.IMREAD_COLOR)
        (h, w) = image_gray.shape[:2]
        if RESIZE != 1:
            image_gray = cv2.resize(image_gray, (h//RESIZE, w//RESIZE))

        if type(image_gray) == type(None):
            print('Open file {} failed.'.format(file))
            continue

        kernel = np.ones((5,5),np.uint8)

        angle = getSurfaceAdjustAngle(image_gray, min_length = 200//RESIZE)

        print('Rotate angle: ', angle)

        print('Before rotation: ', image_gray.shape)
        image = imgRotate2(image_gray, angle)
        print('After rotation: ', image.shape)

        level = getSurfaceLevel(image, min_length = 200//RESIZE)
        print('Surface Level: ', level)

        left_level = int(level[0])
        #print('Left level: ', left_level)

        start = time.time()
        #coreImage = getCoreImage(image, black_limit = 0)
        coreImage = getCoreImage2(lib, image, black_limit = 0)
        lineImage = followCoreLine(lib, coreImage, left_level, min_gap = 50//RESIZE)
        end = time.time()
        print("TIME COST: ", end - start, ' seconds')

        print("MEAN: ", lineImage.max())
        print("\n\n")
        
        #images = np.hstack([image, blur, binary, closed, edages])
        images = np.hstack([image_gray, coreImage, lineImage])

        #np.savetxt('rsmall.csv', image, fmt='%2d', delimiter=',')
        #print(image.max())

        if WRITE_RESULT:
            result_name = file.split('.')[0] + '_res.jpg'
            cv2.imwrite(result_name, coreImage)

        if display.size == 0:
            display = images.copy()
        else:
            display = np.vstack([display, images])

    #display = np.vstack([display])
    cv2.namedWindow('Image', flags = cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Image', 1800, 1000)
    cv2.imshow('Image', display)
    k = cv2.waitKey(0)
    
    cv2.destroyAllWindows()


def main1():
    print("=== Start the WS detecting ===")

    print("=== Read test image ===")

    display = []
    display = np.array(display)

    for file in TEST_IMAGE:
        print('Open file: {}'.format(file))
        image = cv2.imread(file, 0)
        lines_image = cv2.imread(file, cv2.IMREAD_COLOR)

        if type(image) == type(None):
            print('Open file {} failed.'.format(file))
            continue

        image_height = image.shape[0]
        image_width = image.shape[1] 
        mid_line = image_width / 2

        kernel = np.ones((5,5),np.uint8)

        #blur = cv2.fastNlMeansDenoising(image, h = 7.0)
        #blur = cv2.bilateralFilter(image, 5, 21, 21)
        #blur = cv2.GaussianBlur(image, (5, 5), 0)
        blur = cv2.medianBlur(image, 5)
        #opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        #closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        #erosion = cv2.erode(opening, kernel, iterations = 1)
        #ret,binary = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
        ret,binary = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        #binary = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        #binary = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations = 1)
        edages = cv2.Canny(closed, 50, 150, apertureSize = 3)

        #delated = cv2.dilate(edages, kernel, iterations = 1)
        #eroded = cv2.erode(closed, kernel, iterations = 1)

        #lines_image = image.copy()
        """
        lines_image = np.zeros(shape = (*image.shape, 3))
        for i in range(lines_image.shape[0]):
            for j in range(lines_image.shape[1]):
                lines_image[i][j] = [image[i][j], image[i][j], image[i][j]]
        """

        lines_p = cv2.HoughLinesP(closed, 
                                  rho = 1, 
                                  theta = np.pi/180, 
                                  threshold = 100, 
                                  minLineLength = 25, 
                                  maxLineGap = 25)
        
        print('{} lines found. '.format(len(lines_p)))

        #lines = []
        lines_pos = []
        lines_zero_l = []
        lines_zero_r = []
        lines_zero_m = []
        lines_neg = []

        for line in lines_p: 
            one_line = []

            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            theta = np.arctan((y2 - y1) / (x2 - x1))
            theta_app = np.around(theta, 1)

            if theta_app > SLOPE_TH and x2 > mid_line: 
                lines_pos.append([x1, y1, x2, y2, length, theta_app, theta])
            elif theta_app < -SLOPE_TH and x1 < mid_line:
                lines_neg.append([x1, y1, x2, y2, length, theta_app, theta])
            elif theta_app < SLOPE_TH and theta_app > -SLOPE_TH and x2 < mid_line:
                lines_zero_l.append([x1, y1, x2, y2, length, theta_app, theta])
            elif theta_app < SLOPE_TH and theta_app > -SLOPE_TH and x1 > mid_line:
                lines_zero_r.append([x1, y1, x2, y2, length, theta_app, theta])
            #elif theta_app < SLOPE_TH and theta_app > -SLOPE_TH and x1 > mid_line and x2 < mid_line:            
            elif theta_app < SLOPE_TH and theta_app > -SLOPE_TH:
                lines_zero_r.append([x1, y1, x2, y2, length, theta_app, theta])
            
            #print(one_line)
            #lines.append(one_line)
        

        lines_pos = np.array(lines_pos)
        lines_zero_l = np.array(lines_zero_l)
        lines_zero_r = np.array(lines_zero_r)
        lines_zero_m = np.array(lines_zero_m)
        lines_neg = np.array(lines_neg)

        np.set_printoptions(precision=3, suppress=True)
        print("\n\nPositive slope: ")
        print(lines_pos)
        print("\n\nNegative slope: ")
        print(lines_neg)
        print("\n\nZero slope (left): ")
        print(lines_zero_l)
        print("\n\nZero slope (right): ")
        print(lines_zero_r)
        print("\n\nZero slope (middle): ")
        print(lines_zero_m)

        """
        for line in lines_p:
            #print(line)
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
        """

        for line in lines_pos:
            #print(line)
            x1, y1, x2, y2, length, theta_app, theta = line
            cv2.line(lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

        for line in lines_neg:
            #print(line)
            x1, y1, x2, y2, length, theta_app, theta = line
            cv2.line(lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

        for line in lines_zero_l:
            #print(line)
            x1, y1, x2, y2, length, theta_app, theta = line
            cv2.line(lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 1)

        for line in lines_zero_r:
            #print(line)
            x1, y1, x2, y2, length, theta_app, theta = line
            cv2.line(lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 1)

        for line in lines_zero_m:
            #print(line)
            x1, y1, x2, y2, length, theta_app, theta = line
            cv2.line(lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)

        
        #openned = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    
        #images = np.hstack([image, blur, binary, closed, edages])
        images = lines_image

        if display.size == 0:
            display = images.copy()
        else:
            display = np.hstack([display, images])

        if WRITE_RESULT:
            result_name = file.split('.')[0] + '_res.jpg'
            cv2.imwrite(result_name, lines_image)
    
    cv2.namedWindow('Image', flags = cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Image', 1800, 1000)
    cv2.imshow('Image', display)
    k = cv2.waitKey(0)
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()