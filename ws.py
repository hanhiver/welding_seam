import numpy as np 
import cv2 

#TEST_IMAGE = 'ssmall.png'
#TEST_IMAGE = 'sbig.png'
TEST_IMAGE = ('ssmall.png', 'sbig.png', 'rsmall.png', 'rbig.png')

def main():
    print("=== Start the WS detecting ===")

    print("=== Read test image ===")

    display = []
    display = np.array(display)

    for file in TEST_IMAGE:
        image = cv2.imread(file, 0)

        if image.size == 0:
            print('Open file failed. ')
            return

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
        edages = cv2.Canny(binary, 50, 150, apertureSize = 3)
        #delated = cv2.dilate(edages, kernel, iterations = 1)
        #eroded = cv2.erode(closed, kernel, iterations = 1)

        lines_image = edages.copy()
        lines_p = cv2.HoughLinesP(lines_image, 1, np.pi/180, 100, minLineLength = 20, maxLineGap = 5)
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            cv2.line(lines_image, (x1, y1), (x2, y2), (0, 255, 0), 1)

        #openned = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
    
        images = np.hstack([image, blur, binary, edages, lines_image])

        if display.size == 0:
            display = images.copy()
        else:
            display = np.vstack([display, images])

    cv2.imwrite('RESULT.jpg', display)
    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 2800, 1400)
    cv2.imshow('Image', display)
    k = cv2.waitKey(0)


    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()