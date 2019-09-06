import sys
import numpy as np
import cv2
import time

from wslib.BQ_CamMP import BQ_Cam

BOTTOM_THICK

# 共享变量
roi1x, roi1y, roi2x, roi2y = 0, 0, 0, 0           # ROI坐标
point1x, point1y, point2x, point2y = 0, 0, 0, 0   # 鼠标事件坐标
leftButtonDownFlag = False                        # 鼠标释放标志

def on_mouse(event, x, y, flags, param):
    global point1x, point1y, point2x, point2y, leftButtonDownFlag
    global roi1x, roi1y, roi2x, roi2y
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
        roi1x = point1x
        roi1y = point1y
        roi2x = point2x
        roi2y = point2y

def set_bottom_thick(input):
    global BOTTOM_THICK 
    BOTTOM_THICK = input

def main(filename):
    global point1x, point1y, point2x, point2y
    global roi1x, roi1y, roi2x, roi2y

    cam = BQ_Cam(filename)
    
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 1000, 700)
    cv2.moveWindow("result", 100, 100)
    cv2.setMouseCallback('result', on_mouse)
    cv2.createTrackbar('Bottom_Thick','result',20,500,nothing)
    BOTTOM_THICK = cv2.getTrackbarPos('Bottom_Thick','result')

    accum_time = 0
    curr_fps = 0
    prev_time = time.time()
    show_fps = "Show FPS: ??"
    gige_fps = "GigE FPS: ??"
    font_scale = cam.width//800 + 1

    # 初始设置ROI为整幅图像
    roi2x = cam.width
    roi2y = cam.height
    
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

        if (roi1x + roi1y + roi2x + roi2y) > 0:
            cv2.rectangle(frame, pt1=(roi1x, roi1y), pt2=(roi2x, roi2y), 
                          color=(0, 255, 0), thickness=2)

        if leftButtonDownFlag == True and (point1x + point1y + point2x + point2y) > 0:
            cv2.rectangle(frame, pt1=(point1x, point1y), pt2=(point2x, point2y), 
                          color=(0, 0, 255), thickness=2)

        cv2.imshow('result', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1])


