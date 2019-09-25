import os, sys
import numpy as np
import cv2 
import time

# 测试程序。
# python test.py 程序将读取第一个网络摄像头。
# python test.py input.avi 程序将读取视频文件input.avi

def main(filename = None):
    # 初始化frame_provider
    from frame_provider import frame_provider

    if filename is not None: 
        ####################################################
        # 打开文件
        fp = frame_provider(mode = 'file', file = filename)
        ####################################################

    else:
        ####################################################
        # 打开摄像头
        fp = frame_provider(mode = 'cam', cam_ip = "192.168.40.3")
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