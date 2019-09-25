"""
Create on Sep 24 2019.
@author: dhan

适配大恒GigE数字相机python多线程读取程序库。
"""
import os, sys
import numpy as np
import cv2 
import time
import threading


"""
大恒相机或者视频文件多线程读取库
"""
class frame_provider:

    """
    初始化函数
    mode: 
        'file': 文件读取，必须指定文件名。
        'cam': 相机模式，如果不指定相机ip，则默认读取第一个找到的网络相机。
    file: 文件名。
    cam_ip: 相机ip地址。
    auto_expose: 相机是否自动曝光，默认打开。
    width, height: 相机分辨率，默认(1920, 1080)
    """
    def __init__(self, mode = 'file', file = None, cam_ip = None, auto_expose = True, width = 1920, height = 1080):
        
        if mode == 'file':
            self.vid = cv2.VideoCapture(file)
            if not self.vid.isOpened():
                raise IOError("打开文件或本地相机失败: {}".format(input))

            self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.worker = threading.Thread(target = frame_provider.read_file, args = (self, ))
            
        elif mode == 'cam':
            import gxipy as gx 

            self.device_manager = gx.DeviceManager()
            dev_num, dev_info_list = self.device_manager.update_device_list()
            if dev_num == 0 or len(dev_info_list) == 0:
                print('没有找到可用的GigE设备。')
                raise IOError('没有找到可用的GigE设备。')
                
            print("找到{}台设备：{}。".format(dev_num, dev_info_list))

            if cam_ip is None or cam_ip == "": 
                cam_ip = dev_info_list[0].get("ip")
                print("未指定IP，使用： {}".format(dev_info_list[0].get("ip")))
            
            print('打开网络相机 (IP: {}), 进程PID: {}.'.format(cam_ip, os.getpid()))
            self.cam = self.device_manager.open_device_by_ip(cam_ip)
            
            self.cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

            # Set the cam Height and Width. 
            self.cam.Width.set(width)
            self.cam.Height.set(height)

            # Set exposure
            if auto_expose:
                self.cam.ExposureAuto.set(1)
            else:
                self.cam.ExposureAuto.set(0)
                self.cam.ExposureTime.set(200.0)

            # set gain. 
            self.cam.Gain.set(10.0)

            # Start data acquisition. 
            self.cam.stream_on()

            self.width = width
            self.height = height
            self.worker = threading.Thread(target = frame_provider.read_cam, args = (self, ))

        else: 
            raise ValueError('模式设置错误, mode: '.format(mode))

        self.mode = mode 
        self.lock = threading.Lock()
        self.frame = None
        self.fps = 0

    def read_file(self):
        curr_time = time.time()
        prev_time = curr_time
        accum_time = 0
        curr_fps = 0
        
        while True:
            if not self.worker_continue:
                break 
            
            curr_time = time.time()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                with self.lock:
                    self.fps = curr_fps
                curr_fps = 0

            return_value, this_frame = self.vid.read()

            if not return_value: 
                break

            with self.lock:
                self.frame = this_frame
        
        self.vid.release()
        with self.lock:
            self.frame = None
            self.worker_continue = False

    def read_cam(self):
        curr_time = time.time()
        prev_time = curr_time
        accum_time = 0
        curr_fps = 0
        
        while True:
            if not self.worker_continue:
                break 
            
            curr_time = time.time()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                with self.lock:
                    self.fps = curr_fps
                curr_fps = 0

            # Get raw image. 
            raw_image = self.cam.data_stream[0].get_image()
            if raw_image is None:
                print("获取图像失败。")
                continue

            # Get RGB image from raw image.
            rgb_image = raw_image.convert('RGB')
            if rgb_image is None:
                print("Convert raw to RGB failed. ")
                continue

            # Convert RGB image to numpy array. 
            this_frame = rgb_image.get_numpy_array()
            if this_frame is None: 
                print("Convert RGB to np failed. ")
                continue 

            with self.lock:
                self.frame = this_frame
                
        self.cam.stream_off()
        self.cam.close_device()
        with self.lock:
            self.frame = None
            self.worker_continue = False

    """
    开始相机采集或者文件读取线程。
    初始化完成后调用这个函数，采集或者读取真正开始。如果成功开始，函数返回True.
    如果读取线程在0.1s的时间内始终都没能读取到合适的画面，将返回False。
    """
    def start(self):
        self.worker_continue = True 
        self.worker.start()

        for i in range(10):
            if self.frame is None:
                time.sleep(0.01)
            else:
                return True 

        self.worker_continue = False
        return False

    """
    读取一帧。
    如果文件到末尾或者出错，将返回None。
    """
    def read(self):
        with self.lock:
            return self.frame 

    """
    停止图像采集或者文件读取进程。停止之后不可再次开始，除非重新初始化。
    """
    def stop(self):
        self.worker_continue = False
        self.worker.join()


def main(filename = None):
    

    # 初始化frame_provider

    ####################################################
    # 打开文件
    fp = frame_provider(mode = 'file', file = filename)
    ####################################################
    
    ####################################################
    # 打开摄像头
    #fp = frame_provider(mode = 'cam', cam_ip = "192.168.40.3")
    ####################################################

    # 开始文件读取或者相机采集进程。判断是否启动成功。
    ok = fp.start()
    if not ok: 
        print("frame_provider初始化错误。")
        return 

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)

    accum_time = 0
    curr_fps = 0
    prev_time = time.time()
    show_fps = "Show FPS: ??"
    gige_fps = "GigE FPS: ??"
    font_scale = fp.width//800 + 1
    
    while True:          

        # 从相机或者文件读取一帧画面，判断是否出错。
        frame = fp.read()
        if frame is None:
            break 
        
        curr_time = time.time()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            show_fps = "Show FPS: " + str(curr_fps)
            curr_fps = 0
                
        gige_fps = "GigE FPS: " + str(fp.fps)

        cv2.putText(frame, text=gige_fps, org=(30, 80), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=font_scale, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, text=show_fps, org=(30, 160), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=font_scale, color=(0, 0, 255), thickness=2)

        cv2.imshow('result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 停止相机采集或者文件读取进程。
    fp.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()


