import os
import sys
import time
import cv2 as cv 
import numpy as np 
from PIL import Image

import gxipy as gx 

#cam = None
device_manager = None

def initCam():
    #global cam 
    global device_manager

     # Create a device manager. 
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num is 0:
        print('None device found. ')
        return

    print(dev_num)
    for item in dev_info_list:
        print(item)

    # Get the first cam's IP address. 
    cam_ip = dev_info_list[0].get("ip")
    print('Now, open the first cam with IP address {}.'.format(cam_ip))

    # Open the first device. 
    cam = device_manager.open_device_by_ip(cam_ip)

    # Set continues acquisition.
    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

    # Set the cam Height and Width. 
    cam.Width.set(1920)
    cam.Height.set(1200)

    # Set exposure
    cam.ExposureTime.set(10000.0)

    # Set auto white-balance
    cam.BalanceWhiteAuto.set(1)

    # set gain. 
    #cam.Gain.set(10.0)

    # Start data acquisition. 
    cam.stream_on()

    return cam 

def getFrame(cam, time_debug = False):
    #global cam 

    if time_debug:
        time_stamp = time.time()

    # Get raw image. 
    raw_image = cam.data_stream[0].get_image()
    if raw_image is None:
        print("Getting image failed. ")
        return None 

    if time_debug:
        print('Get a frame from cam:   {:3.3f} ms'.format((time.time()-time_stamp) * 1000))
        time_stamp = time.time() 

    # Get RGB image from raw image.
    rgb_image = raw_image.convert('RGB')
    if rgb_image is None:
        print("Convert raw to RGB failed. ")
        return None 
    
    if time_debug:
        print('Convert frame to RGB:   {:3.3f} ms'.format((time.time()-time_stamp) * 1000))
        time_stamp = time.time()

    # Convert RGB image to numpy array. 
    numpy_image = rgb_image.get_numpy_array()
    if numpy_image is None: 
        print("Convert RGB to np failed. ")
        return None 
    
    if time_debug:
        print('Convert frame to numpy: {:3.3f} ms'.format((time.time()-time_stamp) * 1000))
        time_stamp = time.time()

    return numpy_image

def closeCam(cam):
    #global cam 
    # Stop the data aquisition. 
    cam.stream_off()

    # close device. 
    cam.close_device()

def main1():

    # Create a device manager. 
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num is 0:
        print('None device found. ')
        return

    print(dev_num)
    for item in dev_info_list:
        print(item)

    # Get the first cam's IP address. 
    cam_ip = dev_info_list[0].get("ip")
    print('Now, open the first cam with IP address {}.'.format(cam_ip))

    # Open the first device. 
    cam = device_manager.open_device_by_ip(cam_ip)

    # Set continues acquisition.
    cam.TriggerMode.set(gx.GxSwitchEntry.OFF)

    # Set exposure
    #cam.ExposureTime.set(10000.0)
    cam.ExposureTime.set(20000.0)

    # set gain. 
    cam.Gain.set(5.0)

    # Start data acquisition. 
    cam.stream_on()

    cv.namedWindow("result", cv.WINDOW_NORMAL)
    #cv.resizeWindow("result", 640, 400)
    #cv.moveWindow("result", 100, 100)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = time.time()
    time_stamp = time.time()
    frame_stamp = time.time()

    # Acquisition image. 
    while True:
        print('ONE FRAME:  {:3.3f} ms'.format((time.time()-frame_stamp) * 1000))
        frame_stamp = time.time()

        # Get raw image. 
        raw_image = cam.data_stream[0].get_image()
        if raw_image is None:
            print("Getting image failed. ")
            continue
        print('Get a frame from cam:   {:3.3f} ms'.format((time.time()-time_stamp) * 1000))
        time_stamp = time.time() 

        # Get RGB image from raw image.
        rgb_image = raw_image.convert('RGB')
        if rgb_image is None:
            continue 
        print('Convert frame to RGB:   {:3.3f} ms'.format((time.time()-time_stamp) * 1000))
        time_stamp = time.time()

        # Convert RGB image to numpy array. 
        numpy_image = rgb_image.get_numpy_array()
        if numpy_image is None: 
            continue 
        print('Convert frame to numpy: {:3.3f} ms'.format((time.time()-time_stamp) * 1000))
        time_stamp = time.time()

        curr_time = time.time()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv.putText(numpy_image, text=fps, org=(30, 60), fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                    fontScale=2, color=(255, 255, 255), thickness=2)

        cv.imshow('result', numpy_image)
        if cv.waitKey(1) & 0xFF == ord('q'):        
            break

    cv.destroyAllWindows()

    # Stop the data aquisition. 
    cam.stream_off()

    # close device. 
    cam.close_device()


def main():
    #global cam 
    global device_manager

    cam = initCam()

    cv.namedWindow("result", cv.WINDOW_NORMAL)
    cv.resizeWindow("result", 640, 400)
    cv.moveWindow("result", 100, 100)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = time.time()
    time_stamp = time.time()
    frame_stamp = time.time()

    # Acquisition image. 
    while True:
        print('ONE FRAME:  {:3.3f} ms'.format((time.time()-frame_stamp) * 1000))
        frame_stamp = time.time()

        #numpy_image = getFrame()
        numpy_image = getFrame(cam, True)
        #print(numpy_image.shape)

        curr_time = time.time()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv.putText(numpy_image, text=fps, org=(80, 120), fontFace=cv.FONT_HERSHEY_SIMPLEX, 
                    fontScale=4, color=(255, 255, 255), thickness=3)

        cv.imshow('result', numpy_image)
        if cv.waitKey(1) & 0xFF == ord('q'):        
            break

    #closeCam()
    closeCam(cam)

if __name__ == '__main__':
    main()