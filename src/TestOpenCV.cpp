#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>

using namespace std;
using namespace cv;


int main() {

    cout << "OpenCV version: " << CV_VERSION << endl;

    // Open default camera
    VideoCapture cap(0, cv::CAP_V4L2); // Gebruik V4L2 backend (Linux) - pas aan indien nodig voor andere platforms
 
    // Check of camera kan worden geopend ...
    if (!cap.isOpened()) {
        cerr << "Can't open camera" << endl;
        return -1;
    }

     Mat src, hsv, mask1, mask2, mask3, mask;

    while (true) {
        
        // 1. Frame van camera lezen
        cap.read(src);



        double ActualPosition = 0; // Initialize the actual position

        // Display the image
        imshow("Source", src);


        // ESC = stoppen
        if (waitKey(1) == 27) break;
        
    }

    // Camera vrijgeven
    cap.release();

    // Alle vensters sluiten
    destroyAllWindows();

    return 0;
}