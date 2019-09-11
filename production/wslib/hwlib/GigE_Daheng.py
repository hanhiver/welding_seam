"""
Create on Sep 5 2019.
@author: dhan

适配大恒GigE数字相机python程序库。
"""
import os, sys
import multiprocessing
import numpy as np
import cv2 
import time
import ctypes
from multiprocessing.sharedctypes import RawArray, RawValue

VER_MAJOR = 1
VER_MINOR = 0
VER_SPK = 0
"""
返回大恒相机多进程驱动库函数的版本
"""
def version():
    return (VER_MAJOR, VER_MINOR, VER_SPK)


"""
全局共享变量
"""
WIDTH = 1920 
HEIGHT = 1200

device_manager = None        # GigE相机设备管理器
cam = None                   # GigE相机对象
vid = None                   # 图像文件或者本地相机对象

accum_time = 0               # 帧率计算累积时间
curr_fps = 0                 # 当前每秒处理帧数
prev_time = time.time()      # 帧率计算时间戳


"""
获取文件或者摄像头的分辨率。
"""
def get_resolution(filename = None):
    global WIDTH
    global HEIGHT
    
    if filename is None:
        return (WIDTH, HEIGHT)

    vid = cv2.VideoCapture(filename)

    if not vid.isOpened():
        raise IOError("打开文件或本地相机失败: {}".format(input))

    (WIDTH, HEIGHT) = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    vid.release()

    return (WIDTH, HEIGHT)


"""
相机初始化程序
cam_ip: 相机的IP地址。
width, height: 相机分辨率设置
auto_expose: 自动曝光
auto_balance: 自动白平衡
"""
def init_camera(cam_ip = None, width = 1920, height = 1200, auto_expose = True, auto_balance = True):
    import gxipy as gx 
    global cam 
    global device_manager

    try:
        if cam_ip is None:
            # Create a device manager. 
            device_manager = gx.DeviceManager()
            dev_num, dev_info_list = device_manager.update_device_list()
            if dev_num == 0:
                print('没有找到可用设备。')
                return False
        
            # Get the first cam's IP address. 
            cam_ip = dev_info_list[0].get("ip")
            
        # Open the device. 
        print('打开网络相机 (IP: {}), 进程PID: {}.'.format(cam_ip, os.getpid()))
        cam = device_manager.open_device_by_ip(cam_ip)
        print('打开网络相机 (IP: {}), 进程PID: {}.'.format(cam_ip, os.getpid()))
    
        # Set continues acquisition.
        cam.TriggerMode.set(gx.GxSwitchEntry.OFF)
    
        # Set the cam Height and Width. 
        cam.Width.set(width)
        cam.Height.set(height)
    
        # Set exposure
        if auto_expose:
            cam.ExposureAuto.set(1)
        else:
            cam.ExposureAuto.set(0)
            cam.ExposureTime.set(200.0)
    
        # set gain. 
        cam.Gain.set(10.0)
        
        if auto_balance:
            cam.BalanceWhiteAuto.set(1)
        else:
            cam.BalanceWhiteAuto.set(0)
    
        # Start data acquisition. 
        cam.stream_on()
    except Exception as expt:
        print("Error: ", expt)
        return


"""
相机关闭程序
"""
def close_camera():
    import gxipy as gx 
    global cam 
    
    try:
        # Stop the data aquisition. 
        cam.stream_off()
    
        # close device. 
        cam.close_device()
    except Exception as expt:
        print("Error: ", expt)
        return
    
    print("关闭相机。 PID: ", os.getpid())
    return  


"""
从相机读取一帧并写入共享存储区。
shared_array: multiprocessing.sharedctypes.RawArray多进程共享内存空间。
shared_value: multiprocessing.sharedctypes.RawValue多进程共享内存值。
lock: multiprocessing.Lock()进程间同步锁。
time_debug: 是否打印debug信息。
"""
def get_frame_from_camera(shared_array, shared_value, lock, time_debug = False):
    import gxipy as gx 
    global cam 
    global accum_time
    global curr_fps
    global prev_time
    
    curr_time = time.time()
    exec_time = curr_time - prev_time
    prev_time = curr_time
    accum_time = accum_time + exec_time
    curr_fps = curr_fps + 1
    if accum_time > 1:
        accum_time = accum_time - 1
        shared_value.value = curr_fps
        curr_fps = 0
        
    #print("Get frame: ", os.getpid())
    #while True:
    if time_debug:
        time_stamp = time.time()

    # Get raw image. 
    raw_image = cam.data_stream[0].get_image()
    if raw_image is None:
        print("获取图像失败。")
        #continue
        return True

    if time_debug:
        print('Get a frame from cam:   {:3.3f} ms'.format((time.time()-time_stamp) * 1000))
        time_stamp = time.time() 

    # Get RGB image from raw image.
    rgb_image = raw_image.convert('RGB')
    if rgb_image is None:
        print("Convert raw to RGB failed. ")
        #continue
        return True
    
    if time_debug:
        print('Convert frame to RGB:   {:3.3f} ms'.format((time.time()-time_stamp) * 1000))
        time_stamp = time.time()

    # Convert RGB image to numpy array. 
    numpy_image = rgb_image.get_numpy_array()
    if numpy_image is None: 
        print("Convert RGB to np failed. ")
        #continue 
        return True
    
    if time_debug:
        print('Convert frame to numpy: {:3.3f} ms'.format((time.time()-time_stamp) * 1000))
        time_stamp = time.time()

    #print(frame_in_queue.value)
    lock.acquire()
    src = np.ctypeslib.as_ctypes(numpy_image.reshape((HEIGHT * WIDTH * 3)))
    size = ctypes.sizeof(src)
    ctypes.memmove(shared_array, src, size)
    lock.release()
    
    if time_debug:
        print('Send the frame out:     {:3.3f} ms'.format((time.time()-time_stamp) * 1000))
        time_stamp = time.time()
    
    if time_debug:   
        print("")

    return True


"""
初始化读取本地文件或本地相机
filename: 本地文件名(string)或者相机描述符(int)。
"""
def init_file(filename):
    global vid 
    global WIDTH
    global HEIGHT

    vid = cv2.VideoCapture(filename)

    if not vid.isOpened():
        raise IOError("打开文件或本地相机失败: {}".format(input))

    (WIDTH, HEIGHT) = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

"""
从本地文件或者本地相机读取一帧并写入共享存储区。
shared_array: multiprocessing.sharedctypes.RawArray多进程共享内存空间。
shared_value: multiprocessing.sharedctypes.RawValue多进程共享内存值。
lock: multiprocessing.Lock()进程间同步锁。
frame_delay: 每帧图像间隔时间。
time_debug: 是否打印debug信息。
"""
def get_frame_from_file(shared_array, shared_value, lock, frame_delay = 0, time_debug = False):
    global vid  
    global accum_time
    global curr_fps
    global prev_time
    
    curr_time = time.time()
    exec_time = curr_time - prev_time
    prev_time = curr_time
    accum_time = accum_time + exec_time
    curr_fps = curr_fps + 1
    if accum_time > 1:
        accum_time = accum_time - 1
        shared_value.value = curr_fps
        curr_fps = 0
        
    #print("Get frame: ", os.getpid())
    #while True:
    if time_debug:
        time_stamp = time.time()

    return_value, frame = vid.read()

    if return_value: 
        if time_debug:
            print('Get a frame from file:   {:3.3f} ms'.format((time.time()-time_stamp) * 1000))
            time_stamp = time.time() 
        
        if type(frame) != type(None):
            #print(frame_in_queue.value)
            lock.acquire()
            src = np.ctypeslib.as_ctypes(frame.reshape((HEIGHT * WIDTH * 3)))
            size = ctypes.sizeof(src)
            ctypes.memmove(shared_array, src, size)
            lock.release()
        
        if time_debug:
            print('Send the frame out:     {:3.3f} ms'.format((time.time()-time_stamp) * 1000))
            time_stamp = time.time()

        time.sleep(frame_delay)
    
        if time_debug:   
            print("")
    
    else:
        # If reach th EOF or file crashed for reading. 
        lock.acquire()
        shared_value.value = 20000
        lock.release()

    return return_value

"""
关闭本地文件或者本地相机。
"""
def close_file():
    global vid 

    vid.release()
    
    print("Camera closed. PID: ", os.getpid())
    return True


"""
GigE_Daheng模块测试部分
"""
def main1():
    process_lock = multiprocessing.Lock()
    array_temp = np.ones(shape = (HEIGHT * WIDTH * 3), dtype = np.ubyte)
    shared_array = RawArray(ctypes.c_ubyte, array_temp)
    shared_value = RawValue(ctypes.c_uint, 0)

    sched_run = schedrun.SchedRun(func = get_frame_from_camera, args = (shared_array, shared_value, process_lock, False, ), 
                                  init_func = init_camera, init_args = (WIDTH, HEIGHT, ),
                                  clean_func = close_camera, clean_args = {}, 
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
    font_scale = WIDTH//800 + 1
    
    while True:        
        process_lock.acquire()
        frame = np.array(shared_array, dtype = np.uint8)
        process_lock.release()
        frame = frame.reshape((HEIGHT, WIDTH, 3))
        
        curr_time = time.time()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            show_fps = "Show FPS: " + str(curr_fps)
            curr_fps = 0
        
        if shared_value.value > 0:
            gige_fps = "GigE FPS: " + str(shared_value.value)
        
        #fps = gige_fps + " VS " + show_fps
        cv2.putText(frame, text=gige_fps, org=(30, 80), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=font_scale, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, text=show_fps, org=(30, 160), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=font_scale, color=(0, 0, 255), thickness=2)

        cv2.imshow('result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sched_run.stop()
            cv2.destroyAllWindows()
            return True

    # Normally, should not be here. 
    sched_run.stop()
    cv2.destroyAllWindows()
    print('Oops, something wrong')
    return False


def main(filename):
    process_lock = multiprocessing.Lock()
    array_temp = np.ones(shape = (HEIGHT * WIDTH * 3), dtype = np.ubyte)
    shared_array = RawArray(ctypes.c_ubyte, array_temp)
    shared_value = RawValue(ctypes.c_uint, 0)

    sched_run = schedrun.SchedRun(func = get_frame_from_file, args = (shared_array, shared_value, process_lock, ), 
                                  init_func = init_file, init_args = (filename, ),
                                  clean_func = close_file, clean_args = {}, 
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
    font_scale = WIDTH//800 + 1
    
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
        frame = frame.reshape((HEIGHT, WIDTH, 3))
        
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

    sched_run.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.path.append("../pylib")
    import schedrun
    main(sys.argv[1])


