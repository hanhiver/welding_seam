import sys
import numpy as np
import cv2
import time
import ctypes
import multiprocessing
from multiprocessing.sharedctypes import RawArray, RawValue

import wslib.pylib.schedrun as schedrun
import wslib.hwlib.GigE_Daheng as gige 

def main(filename):
    process_lock = multiprocessing.Lock()
    array_temp = np.ones(shape = (gige.HEIGHT * gige.WIDTH * 3), dtype = np.ubyte)
    shared_array = RawArray(ctypes.c_ubyte, array_temp)
    shared_value = RawValue(ctypes.c_uint, 0)

    sched_run = schedrun.SchedRun(func = gige.get_frame_from_file, args = (shared_array, shared_value, process_lock, ), 
                                  init_func = gige.init_file, init_args = (filename, ),
                                  clean_func = gige.close_file, clean_args = {}, 
                                  interval = 0.0, 
                                  init_interval = 0.0)
    
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("result", 800, 500)
    #cv2.moveWindow("result", 100, 100)

    accum_time = 0
    curr_fps = 0
    prev_time = time.time()
    show_fps = "Show FPS: ??"
    gige_fps = "GigE FPS: ??"
    font_scale = gige.WIDTH//800 + 1
    
    while True:        
        if shared_value.value > 10000:
            # Something wrong in the frame aquirement. 
            break
        elif shared_value.value == 0:
            # Fram aquirement process is not ready.
            # sleep for 50ms.  
            sleep(0.05)
            continue

        gige_fps = "GigE FPS: " + str(shared_value.value)

        process_lock.acquire()
        frame = np.array(shared_array, dtype = np.uint8)
        process_lock.release()
        frame = frame.reshape((gige.HEIGHT, gige.WIDTH, 3))
        
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

        cv2.imshow('result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Normally, should not be here. 
    sched_run.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1])


