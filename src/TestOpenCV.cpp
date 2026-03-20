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

    // Load an source image in grayscale.
    Mat src = imread("RodeBal.jpeg", IMREAD_COLOR);

    // 1. BGR → HSV
    Mat hsv;
    cvtColor(src, hsv, COLOR_BGR2HSV);

    // 2. Mask voor rood (2 ranges!)
    Mat mask1, mask2, mask;

    // Lage rood range
    inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), mask1);
    // Hoge rood range
    inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);

    // Combineer
    mask = mask1 | mask2;

    // 3. Ruis verwijderen (optioneel maar sterk aanbevolen)
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
    morphologyEx(mask, mask, MORPH_OPEN, kernel);
    morphologyEx(mask, mask, MORPH_CLOSE, kernel);

    // 4. Contours vinden
    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    

    // 5. Grootste contour pakken (aannemen = bal)
    int largestIndex = 0;
    double maxArea = 0;

    for (int i = 0; i < contours.size(); i++) {

        double area = contourArea(contours[i]);

        if (area > maxArea) {

            maxArea = area;
            largestIndex = i;
        }
    }

    // 6. Middelpunt berekenen (centroid)
    Moments m = moments(contours[largestIndex]);

    int cx = int(m.m10 / m.m00);
    int cy = int(m.m01 / m.m00);

    cout << "Middelpunt: (" << cx << ", " << cy << ")" << endl;

    // 7. Visualisatie
    circle(src, Point(cx, cy), 5, Scalar(0, 255, 0), -1); // Groen cirkeltje op het middelpunt
    drawContours(src, contours, largestIndex, Scalar(255, 0, 0), 2); // Blauwe contour van de bal

    /*

    //blur the image to reduce noise before edge detection.
    Mat src_blurred;
    GaussianBlur(src, src_blurred, Size(3, 3), 0);


    
    // pipeline: thresholding
    Mat dest;
    threshold(src, dest, 32, 255, THRESH_BINARY);

    // 
    Mat canny_output;
    //vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    Canny(dest, canny_output, 50, 150, 3);

    // find contours
    findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

    // get the moments
    vector<Moments> mu(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        mu[i] = moments(contours[i], false);
    }



    // get the centroid of figures.
    vector<Point2f> mc(contours.size());
    for (int i = 0; i < contours.size(); i++)
    {
        mc[i] = Point2f(mu[i].m10 / mu[i].m00, mu[i].m01 / mu[i].m00);
    }

    // draw contours
    Mat drawing(canny_output.size(), CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < contours.size(); i++)
    {
        Scalar color = Scalar(167, 151, 0); // B G R values
        drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
        circle(drawing, mc[i], 10, Scalar(255, 0, 0), -1, 8, 0);
    }

    if (src.empty()) {
        cout << "Could not open or find the image. Please add a sample.jpg to the project root." << endl;
        return -1;
    }

    */

    // Display the image
    imshow("Source", src);
    imshow("HSV", hsv);
    imshow("Mask", mask);
    imshow("kernel", kernel);
    //imshow("Blurred", src_blurred);
    //imshow("Destination", dest);
    //imshow("Canny", canny_output);
    //imshow("Contours", drawing);

    waitKey(0);

    return 0;
}