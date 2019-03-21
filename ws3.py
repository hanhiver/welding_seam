import math
import numpy as np 
import cv2 


#TEST_IMAGE = ('ssmall.png', 'sbig.png', 'rsmall.png', 'rbig.png')
TEST_IMAGE = ('rsmall.png', )


WRITE_RESULT = False
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
    kernel = np.ones((5,5),np.uint8)

    blur = cv2.medianBlur(image, 5)
    ret,binary = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations = 1)

    lines = cv2.HoughLinesP(closed, rho = 1, 
                           theta = np.pi/180, 
                           threshold = 100, 
                           minLineLength = min_length, 
                           maxLineGap = max_line_gap)

    return lines


def getSurfaceAdjustAngle(image, max_angle = 10):
    # Get the dimention of the image and then determine the certer. 
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    lines = getLines(image, min_length = 200)

    zero_slope_lines = []
    max_radian = max_angle * np.pi / 180

    for line in lines: 

        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        theta = np.arctan((y2 - y1) / (x2 - x1))
        theta_apro = np.around(theta, 1)

        if theta_apro < max_radian and theta_apro > -max_radian:

            zero_slope_lines.append([x1, y1, x2, y2, length, theta_apro, theta])

    if zero_slope_lines:
        zero_slope_lines = np.array(zero_slope_lines)

        # Sort the lines with length. 
        index = np.argsort(zero_slope_lines, axis = 0)
        index_length = index[..., 4]
        zero_slope_lines = zero_slope_lines[index_length]

        np.set_printoptions(precision=3, suppress=True)

        # Get the longest X lines:
        x = zero_slope_lines.shape[0] // 4
        print(zero_slope_lines[::-1][:x])
        #for line in zero_slope_lines[::-1][:x]:
        #    print(line)

        ret_radian = np.mean(zero_slope_lines[::-1][:x][..., 6])
        ret_angle = ret_radian * 180 / np.pi 
    else:
        print('Failed to found enough surface lines. ')
        ret_angle = 0

    return ret_angle 
        


def main():
    print("=== Start the WS detecting ===")

    print("=== Read test image ===")

    display = []
    display = np.array(display)

    for file in TEST_IMAGE:
        print('Open file: {}'.format(file))
        image_gray = cv2.imread(file, 0)
        image_color = cv2.imread(file, cv2.IMREAD_COLOR)

        if type(image_gray) == type(None):
            print('Open file {} failed.'.format(file))
            continue


        angle = getSurfaceAdjustAngle(image_gray)

        print('Rotate angle: ', angle)

        #print('Before rotation: ', image_gray.shape)
        image = imgRotate2(image_gray, angle)
        #print('After rotation: ', image.shape)

        #images = np.hstack([image, blur, binary, closed, edages])
        images = image

        if display.size == 0:
            display = images.copy()
        else:
            display = np.hstack([display, images])

        np.savetxt('rsmall.csv', image, fmt='%2d', delimiter=',')

        if WRITE_RESULT:
            result_name = file.split('.')[0] + '_res.jpg'
            cv2.imwrite(result_name, lines_image)
    
    display = np.hstack([image_gray, display])
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