import os, sys, argparse, time, datetime
import queue
import multiprocessing 
import sched
import numpy as np
import cv2 
import time
import ctypes
import schedrun
import gxipy as gx 

# Shared device manager between processes. 
device_manager = None
cam = None

def init_camera():
    global cam 
    global device_manager

     # Create a device manager. 
    device_manager = gx.DeviceManager()
    dev_num, dev_info_list = device_manager.update_device_list()
    if dev_num is 0:
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
    cam.Width.set(1920)
    cam.Height.set(1200)

    # Set exposure
    #cam.ExposureTime.set(1000.0)
    cam.ExposureAuto.set(1)

    # set gain. 
    cam.Gain.set(10.0)
    #cam.GainAuto.set(1)

    cam.BalanceWhiteAuto.set(1)

    # Start data acquisition. 
    cam.stream_on()

def close_camera():
    global cam 
    
    print("Camera closed. ")

    # Stop the data aquisition. 
    cam.stream_off()

    # close device. 
    cam.close_device()

    return
    

def get_frame_from_camera(frame_queue, frame_in_queue, lock, queue_limit = 20, time_debug = False):
    global cam 

    #print("Get frame: ", os.getpid())
    #while True:
    if frame_in_queue.value < queue_limit:
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

        frame_queue.put(numpy_image)
        frame_in_queue.value += 1
        lock.release()

    return True

    #else:
    #    time.sleep(0.05)


def main():
    frame_queue = multiprocessing.Queue()
    frame_in_queue = multiprocessing.Value(ctypes.c_int, 0)
    process_lock = multiprocessing.Lock()

    time_stamp = time.time()

    sched_run = schedrun.SchedRun(func = get_frame_from_camera, args = (frame_queue, frame_in_queue, process_lock, 30, False, ), 
                                  init_func = init_camera, init_args = {},
                                  clean_func = close_camera, clean_args = {}, 
                                  interval = 0.001, 
                                  init_interval = 0)

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 640, 400)
    cv2.moveWindow("result", 100, 100)

    timeout = 1 # Set timeout to 5 seconds. 
    time.sleep(0.5)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = time.time()

    while True:
        try: 
            #frame = frame_queue.get(timeout = timeout)

            #print("SHOW OUT: ", time.time() - time_stamp)
            time_stamp = time.time()
            frame = frame_queue.get(timeout = timeout)

            curr_time = time.time()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(frame, text=fps, org=(30, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                        fontScale=2, color=(255, 255, 255), thickness=3)

            process_lock.acquire()
            frame_in_queue.value -= 1
            process_lock.release()

            cv2.imshow('result', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sched_run.stop()
                cv2.destroyAllWindows()
                return False

        except queue.Empty:
            print('Queue empty.')
            sched_run.stop()
            cv2.destroyAllWindows()
            return False

    sched_run.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()




