#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

#include <iostream>
using namespace std;

extern "C"
{

void readimage(void *buff)
{
    Mat image;
    size_t buf_size; 

    image = imread("./lena.jpg", 0);
    
    buf_size = image.total() * image.channels() * sizeof(char);
    cout<< image.total() << " " << image.channels() << endl;

    //memcpy(image.data, buff, buf_size);
}

int main(int argc, char *argv[])
{
    Mat image; 

    //image = imread("./lena.jpg", 0);
    image = imread("./test.bmp", 0);

    cout<<image.rows<<endl;
    cout<<image.cols<<endl;
    
    for (int i=0; i<image.rows*image.cols; i++)
    {
        cout<<(unsigned int)image.data[i]<<" ";
    }
    cout<<endl;

    void *test = NULL;
    readimage(test);

    //cout<<(unsigned int)image.data[image.rows * image.cols -1]<<endl;
    
    namedWindow("DEMO", 0);

    imshow("DEMO", image);
    
    char key = (char) waitKey(0);
    
    return 0;
}

} // extern "C"
