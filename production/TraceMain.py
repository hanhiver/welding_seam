import sys
import numpy as np
import cv2
import time
import argparse
import logging

from wslib.BQ_CamMP import BQ_Cam
from wslib.BQ_wsPos import BQ_WsPos, PosNormalizer
from wslib.pylib.BQ_imageLib import drawTag, imgRotate
from wslib.pylib.loggerManager import LoggerManager

# 手动设置参数
BOTTOM_THICK = 100     # 初始的底部平台宽度。
NOISY_RATIO = 5        # 底部平台当做噪音滤除的分量，noisy_pixels = botton_thick // NOISY_RATIO
ADJUST_ANGLE = 0       # 根据激光线和相机的安装位置调整初始的图像旋转角度。

# 共享变量
roi1x, roi1y, roi2x, roi2y = 0, 0, 0, 0           # ROI坐标
point1x, point1y, point2x, point2y = 0, 0, 0, 0   # 鼠标事件坐标
leftButtonDownFlag = False                        # 鼠标释放标志
ws = None 
logger_manager = None  # 多进程日志文件记录管理器
logger = None # 本日志文件记录器。
time_stamp = time.time()

def on_mouse(event, x, y, flags, param):
    global point1x, point1y, point2x, point2y, leftButtonDownFlag
    global roi1x, roi1y, roi2x, roi2y
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        leftButtonDownFlag = True
        point1x = x
        point1y = y
    elif event == cv2.EVENT_MOUSEMOVE:         #左键移动
        if(leftButtonDownFlag==True):
            point2x = x
            point2y = y
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        leftButtonDownFlag = False
        point2x = x
        point2y = y
        if (point2x - point1x) > 50 and (point2y - point1y) > 50: 
            roi1x = point1x
            roi1y = point1y
            roi2x = point2x
            roi2y = point2y

def set_bottom_thick(input):
    global ws
    ws.bottom_thick = input
    ws.noisy_pixels = input // NOISY_RATIO

def main(filename, output, arduino = False, log_level = 'warning'):
    global point1x, point1y, point2x, point2y
    global roi1x, roi1y, roi2x, roi2y
    global ws 
    global logger_manager, logger

    logger_manager = LoggerManager(log_level = log_level)

    logger = logger_manager.get_logger('TraceMain')
    logger.info("进入TraceMain主程序。")

    if filename == 'cam': 
        cam = BQ_Cam(logger_manager)
        logger.debug("进入网络相机模式。")
    else:
        cam = BQ_Cam(logger_manager, filename)
        logger.debug("进入文件模式。")

    ws = BQ_WsPos(logger_manager, 
                  bottom_thick = BOTTOM_THICK, 
                  noisy_pixels = BOTTOM_THICK//NOISY_RATIO) 
    logger.debug("初始化WsPos完成。")

    pn = PosNormalizer(logger_manager)
    logger.debug("初始化PosNormalizer完成。")

    # Initialize the arduino serial communication. 
    if arduino is True:
        AS_device = AS.arduino_serial('/dev/ttyUSB0')
        ret = AS_device.openPort()
        if ret is False:
            logger.critical("初始化Arduino失败。")
            return
        logger.debug("初始化Arduino完成。")

    isOutput = True if output != "" else False
    if isOutput:
        output_res = (cam.width, cam.height)

        #video_FourCC = cv2.VideoWriter_fourcc(*'DIVX')
        video_FourCC = cv2.VideoWriter_fourcc("m", "p", "4", "v")
        out = cv2.VideoWriter(output, video_FourCC, 50, output_res)
        out_opened = out.isOpened()
        if out_opened:
            logger.warning('输出文件建立: {}. '.format(output))
        else:
            logger.critical('输出文件建立失败: {}. '.format(output))
            return

    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result", 1000, 700)
    cv2.moveWindow("result", 100, 100)
    cv2.setMouseCallback('result', on_mouse)
    cv2.createTrackbar("BOTTOM", 'result', ws.bottom_thick, 500, set_bottom_thick)
    logger.debug("初始化显示窗口完成。")

    accum_time = 0
    curr_fps = 0
    prev_time = time.time()
    show_fps = "Show FPS: ??"
    gige_fps = "GigE FPS: ??"
    font_scale = cam.width//1000 + 1

    # 初始设置ROI为整幅图像
    roi2x = cam.width
    roi2y = cam.height
    
    while True: 

        time_stamp = frame_stamp = time.time()

        (ok, fps, frame) = cam.read()
        logger.debug("获取一帧图像。") 

        if (logger.getEffectiveLevel() >= logging.INFO):
            time_curr = time.time()
            time_due = (time_curr - time_stamp) * 1000
            time_stamp = time_curr
            logger.info("    {:3.3f} ms 获取一帧图像。".format(time_due)) 
        
        if not ok:
            logger.critical("相机错误或者文件到达末尾。")
            break

        # 根据图像特殊处理
        # ===========================
        frame = imgRotate(frame, ADJUST_ANGLE)
        # ===========================

        roi_image = frame[roi1y:roi2y, roi1x:roi2x]
        logger.debug("按照ROI切割图像，(x1: {}, y1: {}), (x2: {}, y2: {})".format(roi1x, roi1y, roi2x, roi2y))
        
        logger.debug("开始分析ROI图像，bottom_thick: {}, noisy_pixels: {}".format(ws.bottom_thick, ws.noisy_pixels))
        roi_center, roi_level, roi_bound = ws.phaseImage(roi_image)
        logger.debug("分析ROI图像完成， center: {}, level: {}, bound {}".format(roi_center, roi_level, roi_bound))
        
        if (logger.getEffectiveLevel() >= logging.INFO):
            time_curr = time.time()
            time_due = (time_curr - time_stamp) * 1000
            time_stamp = time_curr
            logger.info("    {:3.3f} ms 分析一帧图像。".format(time_due)) 

        frame = ws.fillCoreline2Image(frame, roi1x, roi1y)
        logger.debug("显示图像填充完成，x: {}, y: {}".format(roi_center, roi1x, roi1y))
        
        if (logger.getEffectiveLevel() >= logging.INFO):
            time_curr = time.time()
            time_due = (time_curr - time_stamp) * 1000
            time_stamp = time_curr
            logger.info("    {:3.3f} ms 填充图像轮廓。".format(time_due)) 

        real_center = roi1x + roi_center
        real_level = roi1y + roi_level
        real_bound = (roi_bound[0]+roi1x, roi_bound[1]+roi1x)
        real_center, roi_move = pn.normalizeCenter(real_center)
        logger.debug("输出中点平滑降噪完成，real_center: {}, roi_move: {}".format(real_center, roi_move))

        # Update ROI base on the new center. 
        roi1x_update = roi1x + roi_move
        roi2x_update = roi2x + roi_move
        if roi1x_update > 0 and roi2x_update < cam.width:
            roi1x = roi1x_update
            roi2x = roi2x_update
            logger.debug("ROI窗口自动跟踪移动完成，roi1x: {}, roi2x: {}".format(roi1x, roi2x))

        if (logger.getEffectiveLevel() >= logging.INFO):
            time_curr = time.time()
            time_due = (time_curr - time_stamp) * 1000
            time_stamp = time_curr
            logger.info("    {:3.3f} ms 输出降噪和ROI跟踪。".format(time_due)) 

        # 如果我们开启了arduino serial通讯，这里拼凑坐标传送给机器人。
        if arduino is True:

            # 这个地方的real_center就是焊缝识别得到的中点坐标。
            # 我们在这里用传统380线分辨率的做一个规一划处理。
            # output_center = real_center // 3
        
            ####################################
            # 这个地方请修改代码，是否应该将real_center的值转化为16进制的数字，
            # 通信的高地位分别怎么设置，我这里就只是用了你示例代码中的通信标志。
            ####################################
            AS_device.writePort('E7E701450124005A0008004EFE')
            time_curr = time.time()
            time_due = (time_curr - time_stamp) * 1000
            time_stamp = time_curr
            logger.info("    {:3.3f} ms 坐标写入串口完成。".format(time_due))

        drawTag(frame, real_center, real_level, bound = real_bound, bottom_thick = ws.bottom_thick)
        logger.debug("输出图像标记完成。")

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
                
        #fps = gige_fps + " VS " + show_fps
        cv2.putText(frame, text=gige_fps, org=(30, 80), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=font_scale, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, text=show_fps, org=(30, 160), fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=font_scale, color=(0, 0, 255), thickness=2)

        if (roi1x + roi1y + roi2x + roi2y) > 0:
            cv2.rectangle(frame, pt1=(roi1x, roi1y), pt2=(roi2x, roi2y), 
                          color=(0, 255, 0), thickness=2)

        if leftButtonDownFlag == True and (point1x + point1y + point2x + point2y) > 0:
            cv2.rectangle(frame, pt1=(point1x, point1y), pt2=(point2x, point2y), 
                          color=(0, 0, 255), thickness=2)

        if (logger.getEffectiveLevel() >= logging.INFO):
            time_curr = time.time()
            time_due = (time_curr - time_stamp) * 1000
            time_stamp = time_curr
            logger.info("    {:3.3f} ms 输出图像标记。".format(time_due)) 

        cv2.imshow('result', frame)
        logger.debug("图像输出屏幕完成。")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.debug("收到手动退出指令。")
            break

        if (logger.getEffectiveLevel() >= logging.INFO):
            time_curr = time.time()
            time_due = (time_curr - time_stamp) * 1000
            time_stamp = time_curr
            logger.info("    {:3.3f} ms 图像屏幕输出。".format(time_due)) 

            time_due = (time.time() - frame_stamp) * 1000
            logger.info("{:3.3f} ms 本帧图像处理完成。".format(time_due))

    logger.debug("退出TraceMain主程序，销毁所有显示窗口。")
    logger_manager.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    #print(sys.argv[1])
    parser = argparse.ArgumentParser()

    # 输出文件。
    parser.add_argument('-o', '--output', type = str, default = '', 
                        help = '[Optional] Output video. ')

    # 是否连接Arduino_serial进行通讯。
    parser.add_argument('-a', '--arduino', default = False, action = "store_true", 
                        help = '[Enable the Arduino Serial Communication. ')
    
    # 日志级别
    parser.add_argument('-l', '--loglevel', type = str, default = 'warning',
                        help = '[Optional] Log level. WARNING is default. ')

    # 默认处理所有文件选项。
    parser.add_argument('input', type = str, default = None, nargs = '+',
                        help = 'Input files. ')

    FLAGS = parser.parse_args()

    main(FLAGS.input[0],  
         output = FLAGS.output,
         arduino = FLAGS.arduino,
         log_level = FLAGS.loglevel, 
         )


