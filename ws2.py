import math
import numpy as np 
import cv2 

TEST_IMAGE = ('ssmall.png', 'sbig.png', 'rsmall.png', 'rbig.png')
#TEST_IMAGE = ('rbig.png', )

WRITE_RESULT = True
SLOPE_TH = 0.15


def main():
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
        cv2.imwrite('RESULT.jpg', display)
    
    cv2.namedWindow('Image', flags = cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('Image', 1800, 1000)
    cv2.imshow('Image', display)
    k = cv2.waitKey(0)
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()