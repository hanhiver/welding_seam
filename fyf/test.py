import os, sys
import threading
import threading
import numpy as np
import cv2 
import time

# Global         
frame = None
thread_continue = True
thread_fps = 0

def read_cam_thread(vid, lock):
    global frame 
    global thread_continue 
    global thread_fps

    curr_time = time.time()
    prev_time = curr_time
    accum_time = 0
    curr_fps = 0

    while True:
        if not thread_continue:
            break 

        curr_time = time.time()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            thread_fps = curr_fps
            curr_fps = 0

        return_value, this_frame = vid.read()
        if not return_value:
            frame = None 
            break

        with lock:
            frame = this_frame


def main(filename):
    global frame
    vid = cv2.VideoCapture(filename)

    if not vid.isOpened():
        raise IOError("打开文件或本地相机失败: {}".format(input))
    (WIDTH, HEIGHT) = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("result", 800, 500)
    #cv2.moveWindow("result", 100, 100)

    accum_time = 0
    curr_fps = 0
    prev_time = time.time()
    show_fps = "Show FPS: ??"
    gige_fps = "GigE FPS: ??"
    font_scale = WIDTH//800 + 1

    lock = threading.Lock()
    cam_thread = threading.Thread(target = read_cam_thread, args = (vid, lock, ))
    cam_thread.start()

    for i in range(10):
        if frame is None:
            time.sleep(0.01)

    if frame is None:
        print("Read thread error. ")
        return

    while True:        

        #gige_fps = "GigE FPS: " + str(shared_value.value)
        gige_fps = "GigE FPS: " + str(thread_fps)

        """
        frame = read_cam_thread(vid)
        if frame is None:
            raise IOError("文件到达末尾或者出错. ")
        """

        """
        return_value, frame = vid.read()
        if not return_value:
            raise IOError("文件到达末尾或者出错. ")
        """

        #frame = frame.reshape((HEIGHT, WIDTH, 3))
        
        curr_time = time.time()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            show_fps = "Show FPS: " + str(curr_fps)
            curr_fps = 0

        with lock:
            this_frame = frame

        if this_frame is None:
            thread_continue = False
            break
                
        #fps = gige_fps + " VS " + show_fps
        cv2.putText(this_frame, text=gige_fps, org=(30, 80), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=font_scale, color=(0, 0, 255), thickness=2)
        cv2.putText(this_frame, text=show_fps, org=(30, 160), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=font_scale, color=(0, 0, 255), thickness=2)

        cv2.imshow('result', this_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            thread_continue = False
            break

    cv2.destroyAllWindows()
    cam_thread.join()
    vid.release() 


if __name__ == '__main__':
    main(sys.argv[1])


