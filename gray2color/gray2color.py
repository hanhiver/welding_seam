import numpy as np
import cv2 as cv

K_SIZE = 6

def main(input='./test.mp4', output='./res.avi', wb=False):
    vid = cv.VideoCapture(input)

    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video: {}".format(input))
    
    #video_FourCC    = int(vid.get(cv.CAP_PROP_FOURCC))
    video_fps       = vid.get(cv.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv.CAP_PROP_FRAME_WIDTH)),
                    int(vid.get(cv.CAP_PROP_FRAME_HEIGHT))*2)
    video_FourCC = cv.VideoWriter_fourcc(*'MP42')
    if wb:
        out = cv.VideoWriter(output, video_FourCC, video_fps, video_size)
        if not out.isOpened():
            print("Open output file failed. ")
            exit(-1)

    while True:
        ret, frame = vid.read()
        
        cv.namedWindow("Result")

        if ret: 
            (h, w) = frame.shape[:2]
            #gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            (b, g, r) = cv.split(frame)
            gray = r
            conv = cv.filter2D(gray, -1, np.ones([K_SIZE, K_SIZE])/(K_SIZE*K_SIZE))
            (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(conv)

            #gray = cv.equalizeHist(gray)
            color = cv.applyColorMap(gray, cv.COLORMAP_JET)
            cv.circle(color, maxLoc, K_SIZE+1, (255, 255, 0), 1)

            show_img = np.vstack([frame, color])
            if wb:
                out.write(show_img)
            
            cv.imshow("Result", show_img)
            if cv.waitKey(100) & 0xFF == ord('q'):
                break
        else:
            break
    
    if wb:
        out.release()
    
    cv.destroyAllWindows()

if __name__ == '__main__':
    print(cv.__version__)
    main(wb=True)




