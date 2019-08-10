#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 16:12:06 2019

@author: dhan
"""
import os, sys
import multiprocessing
import numpy as np
import cv2 
import time
import ctypes
import schedrun
import gxipy as gx 
from multiprocessing.sharedctypes import RawArray, RawValue

WIDTH = 1920 
HEIGHT = 1200

# Shared device manager between processes. 
device_manager = None
cam = None
accum_time = 0
curr_fps = 0
prev_time = time.time()

def init_camera(width = 1920, height = 1200, auto_expose = True, auto_balance = True):
    global cam 
    global device_manager

     # Create a device manager. 
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num == 0:
        print('None device found. ')
        return

    # Get the first cam's IP address. 
    cam_ip = dev_info_list[0].get("ip")
    print('Now, open the first cam with IP address {}.'.format(cam_ip))

    # Open the first device. 
    cam = device_manager.open_device_by_ip(cam_ip)

    # Set continues acquisition.
    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

    # Set the cam Height and Width. 
    cam.Width.set(width)
    cam.Height.set(height)

    # Set exposure
    if auto_expose:
        cam.ExposureAuto.set(1)
    else:
        cam.ExposureTime.set(1000.0)

    # set gain. 
    cam.Gain.set(10.0)
    #cam.GainAuto.set(1)
    
    if auto_balance:
        cam.BalanceWhiteAuto.set(1)

    # Start data acquisition. 
    cam.stream_on()
    
    return cam

def close_camera():
    global cam 
    
    print("Camera closed. ")

    # Stop the data aquisition. 
    cam.stream_off()

    # close device. 
    cam.close_device()

    return
    
def get_frame_from_camera(shared_array, shared_value, lock, time_debug = False):
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
        print("Getting image failed. ")
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

    return True


def main():
    process_lock = multiprocessing.Lock()
    array_temp = np.ones(shape = (HEIGHT * WIDTH * 3), dtype = np.ubyte)
    shared_array = RawArray(ctypes.c_ubyte, array_temp)
    shared_value = RawValue(ctypes.c_uint, 0)

    sched_run = schedrun.SchedRun(func = get_frame_from_camera, args = (shared_array, shared_value, process_lock, False, ), 
                                  init_func = init_camera, init_args = (WIDTH, HEIGHT, ),
                                  clean_func = close_camera, clean_args = {}, 
                                  interval = 0.001, 
                                  init_interval = 0)
    
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 800, 500)
    cv2.moveWindow("result", 100, 100)

    accum_time = 0
    curr_fps = 0
    prev_time = time.time()
    show_fps = "SHOW FPS: ??"
    gige_fps = "GigE FPS: ??"
    
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
        
        fps = gige_fps + " VS " + show_fps
        cv2.putText(frame, text=fps, org=(30, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale=2, color=(0, 0, 255), thickness=4)

        cv2.imshow('result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            sched_run.stop()
            cv2.destroyAllWindows()
            return False

    sched_run.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()




