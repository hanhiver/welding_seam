import os, sys, argparse, time, datetime
import queue
import multiprocessing 
import sched
import numpy as np
import cv2 
import time
import ctypes
#from PIL import Image

# 多进程视频文件摄像头读取测试程序。
# python schedrun.py: 默认打开0号摄像头测试。
# python schedrun.py test.avi: 打开test.avi视频文件测试。


# Class for multithreading open another task with certain interval. 
class SchedRun():

    # func: worker function that will call after each interval time. 
    # args: argument of the worker function. 
    # init_func (optional): initialization function. 
    # init_args (optional): initialization function argument.
    # interval: the func will be call after each interval of time. 
    # init_interval: the interval between init call and real worker function. 
    def __init__(self, 
                 func, args, 
                 init_func = None, init_args = {}, 
                 clean_func = None, clean_args = {}, 
                 interval = 0.04, init_interval = 0.5):

        self.func = func
        self.args = args
        self.init_func = init_func
        self.init_args = init_args
        self.clean_func = clean_func
        self.clean_args = clean_args

        self.interval = interval
        self.init_interval = init_interval
        self.event_id = None

        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.sub_process = multiprocessing.Process(target = self.wrap_process, args = {}, daemon = True)
        self.sub_process_continue = multiprocessing.Value(ctypes.c_bool, True)
        self.sub_process.start()

    def __del__(self):
        pass
        #self.sub_process.join()
        #print('SchedRun terminated. ')

    def wrap_func(self):
        if self.sub_process_continue.value:
            self.scheduler.enter(self.interval, 1, self.wrap_func)
            #self.sub_process_continue.value = self.func(*self.args)
            self.func(*self.args)
        else:
            self.scheduler.enter(0.0, 1, self.clean_func, argument = self.clean_args)
            return

    def wrap_process(self):
        if self.init_func:
            res = self.init_func(*self.init_args)
            if res:
                print("Initilization Failed.")
                self.sub_process_continue = False
                return

        self.scheduler.enter(self.init_interval, 1, self.wrap_func) 
        self.scheduler.run()

        #self.clean_func(*self.init_args)
        return

    def stop(self):
        self.sub_process_continue.value = False

        if self.sub_process.is_alive():
            print("Join Process")
            self.sub_process.join(1)
            print("Clean Process")
            self.sub_process.terminate()

vid = None

def init_camera(cam_input):
    global vid
    
    if cam_input:
        vid = cv2.VideoCapture(cam_input)
    else:
        vid = cv2.VideoCapture(0)

    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    video_FourCC    = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    print("INIT PID: ", os.getpid())

    #print("!!! TYPE:", type(cam_input), type(video_FourCC), type(video_fps), type(video_size))
    #print("!!! TYPE:", cam_input, video_FourCC, video_fps, video_size)

def get_frame_from_camera(frame_queue, frame_in_queue, lock, queue_limit = 20):
    global vid 

    print("WORKER PID", os.getpid())
    time_stamp = time.time()

    while True:
        if frame_in_queue.value < queue_limit:
            print("READ OUT: ", time.time() - time_stamp)
            time_stamp = time.time()
            got_a_frame, frame = vid.read()

            #print(frame_in_queue.value)
            lock.acquire()

            if type(frame) != type(None):
                frame_queue.put(frame)
                frame_in_queue.value += 1
                lock.release()
                #time.sleep(0.05)
                #print("READ IN: ", time.time() - time_stamp)
            else:
                print('Video Finished.')
                lock.release()
                vid.release()
                break
        else:
            time.sleep(0.05)


def main(input_file):
    frame_queue = multiprocessing.Queue()
    frame_in_queue = multiprocessing.Value(ctypes.c_int, 0)
    process_lock = multiprocessing.Lock()

    time_stamp = time.time()

    sched_run = SchedRun(func = get_frame_from_camera, args = (frame_queue, frame_in_queue, process_lock, 30, ), 
                         init_func = init_camera, init_args = (input_file, ),
                         interval = 0.01, 
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
                        fontScale=2, color=(255, 255, 255), thickness=2)

            process_lock.acquire()
            frame_in_queue.value -= 1
            process_lock.release()

            cv2.imshow('result', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                sched_run.stop()
                cv2.destroyAllWindows()
                return False

            #print("SHOW IN: ", time.time() - time_stamp)
            

        except queue.Empty:
            print('Queue empty.')
            sched_run.stop()
            cv2.destroyAllWindows()
            return False

    cv2.destroyAllWindows()

if __name__ == '__main__':
    #main('./1out.avi')
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main(None)




