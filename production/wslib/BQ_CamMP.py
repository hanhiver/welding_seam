import sys
import numpy as np
import cv2
import time
import ctypes
import multiprocessing
from multiprocessing.sharedctypes import RawArray, RawValue

sys.path.append("..")
import wslib.pylib.schedrun as schedrun
import wslib.hwlib.GigE_Daheng as gige 

"""
BQ_Cam提供了多进程打开文件或者相机功能。
"""
class BQ_Cam():
    """
    相机初始化。
    cam_ip: 相机的IP地址。
    filename: 目标视频文件。
        如果filename为None，则选择打开相机模式，自动打开cam_ip所指网络相机。
        如果cam_ip为空，自动选择列表中第一台网络相机打开。

    width, height: 相机模式下，设置相机分辨率。文件模式下无效。
    auto_expose: 相机自动曝光。文件模式下无效。
    auto_balance: 相机自动白平衡，文件模式下无效。
    """
    def __init__(self, logger_manager, 
                 filename = None, cam_ip = None,  
                 width = 1920, height = 1200, 
                 auto_expose = True, auto_balance = True):
        
        self.logger = logger_manager.get_logger("BQ_Cam")
        self.sched_run = None
        if filename is None:
            # 设置相机模式标识。
            self.mode = 0
            # 相机模式分辨率自设，默认1920*1200
            (self.w, self.h) = (width, height)
            self.cam_ip = cam_ip
            self.auto_expose = auto_expose
            self.auto_balance = auto_balance
            self.logger.debug("网络相机模式，IP: {}, (w, h): ({}, {}).".format(cam_ip, self.w, self.h))
        else:
            # 设置文件模式标识。
            self.mode = 1
            # 文件模式获取分辨率
            (self.w, self.h) = gige.get_resolution(filename)
            self.logger.debug("网络相机模式，filename: {}, (w, h): ({}, {}).".format(filename, self.w, self.h))

        self.width = self.w
        self.height = self.h

        # 准备进程锁和进程间共享空间。
        self.process_lock = multiprocessing.Lock()
        self.array_temp = np.ones(shape = (self.h * self.w * 3), dtype = np.ubyte)
        self.shared_array = RawArray(ctypes.c_ubyte, self.array_temp)
        self.shared_value = RawValue(ctypes.c_uint, 0)
        self.logger.debug("进程共享内存准备完毕。")

        if self.mode == 0: 
            self.sched_run = schedrun.SchedRun(
                        func = gige.get_frame_from_camera, args = (self.shared_array, self.shared_value, self.process_lock, False, ), 
                        init_func = gige.init_camera, init_args = (self.cam_ip, self.w, self.h, self.auto_expose, self.auto_balance),
                        clean_func = gige.close_camera, clean_args = {}, 
                        interval = 0.0, 
                        init_interval = 0.0)
        else:
            self.sched_run = schedrun.SchedRun(
                        func = gige.get_frame_from_file, args = (self.shared_array, self.shared_value, self.process_lock, ), 
                        init_func = gige.init_file, init_args = (filename, ),
                        clean_func = gige.close_file, clean_args = {}, 
                        interval = 0.025, 
                        init_interval = 0.0)

        self.logger.debug("帧读取进程启动。")

        #while self.shared_value != 0:

        for i in range(10):
            if self.shared_value == 0:    
                self.logger.debug("帧读取进程没准备好，等待100ms。")
                # 相机进程还没有准备好。
                time.sleep(0.1)

        if self.shared_value == 0:
            return None


    def ready(self):
        return self.sched_run.ready()

    """
    从相机获取一帧图像。
    返回值包含三部分:
        ok: 成功标志，如果标志为False，相机出错或者文件到末尾。
        fps: 相机读取帧率。(一定会<10000fps, 出错情况下为None)
        img: 获取到的图像。(出错情况下为None)
    """
    def read(self):
        if self.shared_value.value > 10000:
            self.logger.debug("相机进程错误或文件到达末尾。shared_value: {}".format(self.shared_value.value))
            # 相机进程出错或者相机文件读取到末尾
            return (False, None, None)
    
        self.process_lock.acquire()
        frame = np.array(self.shared_array, dtype = np.uint8)
        fps = self.shared_value.value
        self.process_lock.release()
        frame = frame.reshape((self.h, self.w, 3))

        return (True, fps, frame)

    """
    相机结束的时候自动清理进程关闭相机。
    """
    def __del__(self):
        if self.sched_run != None:
            self.sched_run.stop()

"""
BQ_CamMP模块测试函数。
"""
def main(filename):
    cam = BQ_Cam(filename)
    
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("result", 800, 500)
    #cv2.moveWindow("result", 100, 100)

    accum_time = 0
    curr_fps = 0
    prev_time = time.time()
    show_fps = "Show FPS: ??"
    gige_fps = "GigE FPS: ??"
    font_scale = cam.width//800 + 1
    clahe = cv2.createCLAHE(clipLimit = 40, tileGridSize = (8, 8))
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

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        #frame = cv2.equalizeHist(frame)
        clahe.apply(frame)
                
        #fps = gige_fps + " VS " + show_fps
        cv2.putText(frame, text=gige_fps, org=(30, 80), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=font_scale, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, text=show_fps, org=(30, 160), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=font_scale, color=(0, 0, 255), thickness=2)

        cv2.imshow('result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1])

