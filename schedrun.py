import os, sys, argparse, time, datetime
import queue
import multiprocessing 
import sched
import numpy as np
import cv2 
import time
#from PIL import Image

# Class for multithreading open another task with certain interval. 
class SchedRun():

    # func: worker function that will call after each interval time. 
    # args: argument of the worker function. 
    # init_func (optional): initialization function. 
    # init_args (optional): initialization function argument.
    # interval: the func will be call after each interval of time. 
    # init_interval: the interval between init call and real worker function. 
    def __init__(self, func, args, init_func = None, init_args = {}, interval = 0.04, init_interval = 0.5):
        self.func = func
        self.args = args
        self.init_func = init_func
        self.init_args = init_args
        self.interval = interval
        self.init_interval = init_interval
        self.event_id = None

        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.sub_process = multiprocessing.Process(target = self.wrap_process, args = {}, daemon = True)
        self.sub_process_continue = True
        self.sub_process.start()

    def __del__(self):
        self.sub_process.join()
        #print('SchedRun terminated. ')

    def wrap_func(self):
        if self.sub_process_continue:
            self.scheduler.enter(self.interval, 1, self.wrap_func)
            self.sub_process_continue = self.func(*self.args)
        else:
            #print('wrap_func finished. ')
            return

    def wrap_process(self):
        if self.init_func:
            self.init_func(*self.init_args)

        self.event_id = self.scheduler.enter(self.init_interval, 1, self.wrap_func) 
        self.scheduler.run()
        #print('wrap_process finished. ')
        return

    def stop(self):
        if self.event_id is not None:
            self.scheduler.cancel(self.event_id)
        
        if self.sub_process.is_alive():
            self.sub_process.terminate()

        self.event_id = None

vid = None

def init_camera(cam_input):
    global vid
    vid = cv2.VideoCapture(cam_input)

    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    print("!!! TYPE:", type(cam_input), type(video_FourCC), type(video_fps), type(video_size))
    print("!!! TYPE:", cam_input, video_FourCC, video_fps, video_size)

def get_frame_from_camera(frame_queue):
    global vid 

    while True:
        got_a_frame, frame = vid.read()
        if type(frame) != type(None):
            frame_queue.put(frame)
            #time.sleep(0.05)
        else:
            print('Video Finished.')
            break

def main(input_file):
    frame_queue = multiprocessing.Queue()

    sched_run = SchedRun(func = get_frame_from_camera, args = {frame_queue}, 
                         init_func = init_camera, init_args = {input_file},
                         interval = 0.01, 
                         init_interval = 0.05)

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 640, 400)
    cv2.moveWindow("result", 100, 100)

    timeout = 1 # Set timeout to 5 seconds. 
    while True:
        try:
            frame = frame_queue.get(timeout = timeout)

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

    cv2.destroyAllWindows()

if __name__ == '__main__':
    #main('./1out.avi')
    main('/Users/dhan/upload/fm1.mp4')




