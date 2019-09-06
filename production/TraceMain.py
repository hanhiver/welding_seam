import sys
import numpy as np
import cv2
import time

from wslib.BQ_CamMP import BQ_Cam

point1x = 0
point1y = 0
point2x = 0
point2y = 0
leftButtonDownFlag = False

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


def nothing(input):
    print("Set to: ", input)

def main(filename):
    global point1x,point1y,point2x,point2y

    cam = BQ_Cam(filename)
    
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 1000, 700)
    cv2.moveWindow("result", 100, 100)
    cv2.setMouseCallback('result', on_mouse)
    cv2.createTrackbar('Bottom_Thick','result',20,500,nothing)
    BOTTOM_THICK = cv2.getTrackbarPos('Bottom_Thick','result')
    #showCrosshair = False
    #fromCenter = False
    #r = cv2.selectROI("Image", frame, fromCenter, showCrosshair)
    #print(r)

    accum_time = 0
    curr_fps = 0
    prev_time = time.time()
    show_fps = "Show FPS: ??"
    gige_fps = "GigE FPS: ??"
    font_scale = cam.width//800 + 1
    
    while True: 
        (ok, fps, frame) = cam.read()       
        
        if not ok:
            print("Cam Error or file EOF. ")
            break

        gige_fps = "GigE FPS: " + str(fps)
        
        curr_time = time.time()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            show_fps = "Show FPS: " + str(curr_fps)
            curr_fps = 0
                
        #fps = gige_fps + " VS " + show_fps
        cv2.putText(frame, text=gige_fps, org=(30, 80), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=font_scale, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, text=show_fps, org=(30, 160), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=font_scale, color=(0, 0, 255), thickness=2)

        if (point1x + point1y + point2x + point2y) > 0:
            cv2.rectangle(frame, pt1 = (point1x, point1y), pt2 = (point2x, point2y), 
                          color = (0, 255, 0), thickness = 2)

        cv2.imshow('result', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1])


