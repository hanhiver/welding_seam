#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

#include <iostream>
#include <ctime>

define CAM_MAX_TESTED 10

clock_t time_stamp;

/*
 * Get the index of first valide camera in the system. 
 * maxTested: the max index that will test in the func. 
 * return: 
 *     success: the index number of the first valide cam. 
 *     failed: -1. 
 */
int getFirstCamIndex(int maxTested)
{
    for (int i=0; i<maxTested; i++)
    {
        VideoCapture temp(i);

        bool res = temp.isOpened();
        temp.release();

        if (res)
        {
            return i;
        }
    }

    return -1;
}

class UsbCamera
{
public:
    UsbCamera();
    UsbCamera(int camIndex);

    ~UsbCamera();
    bool camReady();
    int read(cv::Mat srcImg);

private:
    cv::VideoCapture _cap; 
    bool _ready;
};

UsbCamera::UsbCamera()
{
    _ready = false;
}

UsbCamera::UsbCamera(int camIndex)
{
    _cap = cv::VideoCapture(camIndex);
    _ready = true;
}

~UsbCamera::UsbCamera()
{
    _ready = false;
    _cap.release()
}

bool UsbCamera::camReady()
{
    return _ready;
}

int UsbCamera::read(cv::Mat srcImg)
{
    _cap.read(srcImg);
}

int main()
{
    return 0;
}

