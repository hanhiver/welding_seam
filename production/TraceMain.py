import sys
import numpy as np
import cv2
import time

from wslib.BQ_CamMP import BQ_Cam
from wslib.BQ_wsPos import BQ_WsPos, PosNormalizer
from wslib.pylib.BQ_imageLib import drawTag

# 共享变量
roi1x, roi1y, roi2x, roi2y = 0, 0, 0, 0           # ROI坐标
point1x, point1y, point2x, point2y = 0, 0, 0, 0   # 鼠标事件坐标
leftButtonDownFlag = False                        # 鼠标释放标志
ws = None 

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
        if (point2x - point1x) > 100 and (point2y - point1y) > 100: 
            roi1x = point1x
            roi1y = point1y
            roi2x = point2x
            roi2y = point2y

def set_bottom_thick(input):
    global ws
    ws.bottom_thick = input

def main(filename):
    global point1x, point1y, point2x, point2y
    global roi1x, roi1y, roi2x, roi2y
    global ws 

    cam = BQ_Cam(filename)
    ws = BQ_WsPos() 
    pn = PosNormalizer()
    ws.testlib()
    
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 1000, 700)
    cv2.moveWindow("result", 100, 100)
    cv2.setMouseCallback('result', on_mouse)
    cv2.createTrackbar('Bottom_Thick', 'result', ws.bottom_thick, 500, set_bottom_thick)
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

        roi_image = frame[roi1y:roi2y, roi1x:roi2x]
        #roi_image = cv2.normalize(roi_image,dst=None,alpha=350,beta=10,norm_type=cv2.NORM_MINMAX)

        roi_center, roi_level, roi_bound = ws.phaseImage(roi_image)
        frame = ws.fillCoreline2Image(frame, roi1x, roi1y)
        
        real_center = roi1x + roi_center
        real_level = roi1y + roi_level
        real_bound = (roi_bound[0]+roi1x, roi_bound[1]+roi1x)
        real_center, roi_move = pn.normalizeCenter(real_center)

        # Update ROI base on the new center. 
        roi1x_update = roi1x + roi_move
        roi2x_update = roi2x + roi_move
        if roi1x_update > 0 and roi2x_update < cam.width:
            roi1x = roi1x_update
            roi2x = roi2x_update

        drawTag(frame, real_center, real_level, bound = real_bound)

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


