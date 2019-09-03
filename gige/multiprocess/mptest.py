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
from multiprocessing.sharedctypes import RawArray, RawValue

import gxipy as gx 
import schedrun

WIDTH = 1920 
HEIGHT = 1200

# Shared device manager between processes. 
device_manager = None
cam = None

# Shared vid to read video file or open local cam. 
vid = None

accum_time = 0
curr_fps = 0
prev_time = time.time()

def init_camera(width = 1920, height = 1200, auto_expose = True, auto_balance = True):
    global cam 
    global device_manager

    try:
        # Create a device manager. 
        device_manager = gx.DeviceManager()
        dev_num, dev_info_list = device_manager.update_device_list()
        if dev_num == 0:
            print('None device found. ')
            return False
    
        # Get the first cam's IP address. 
        cam_ip = dev_info_list[0].get("ip")
        print('Open camera (IP: {}), PID: {}.'.format(cam_ip, os.getpid()))
    
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
    

def close_camera():
    global cam 
    
    try:
        # Stop the data aquisition. 
        cam.stream_off()
    
        # close device. 
        cam.close_device()
    except Exception as expt:
        print("Error: ", expt)
        return
    
    print("Camera closed. PID: ", os.getpid())
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
    
    if time_debug:
        print('Send the frame out:     {:3.3f} ms'.format((time.time()-time_stamp) * 1000))
        time_stamp = time.time()
    
    if time_debug:   
        print("")

    return True


def init_file(filename):
    global vid 

    vid = cv2.VideoCapture(filename)

    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video: {}".format(input))

def get_frame_from_file(shared_array, shared_value, lock, frame_delay = 0.035, time_debug = False):
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

    return return_value

def close_file():
    global vid 
    
    print("Camera closed. PID: ", os.getpid())
    return


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


if __name__ == '__main__':
    main(sys.argv[1])




