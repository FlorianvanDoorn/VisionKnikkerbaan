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

        // 1. BGR → HSV
        cvtColor(src, hsv, COLOR_BGR2HSV);

        // 2. Mask voor rood (2 ranges!)
        inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), mask1);  // Lage rood range
        inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), mask2);   // Hoge rood range

        // inRange(hsv, Scalar(0, 0, 215), Scalar(180, 80, 255), mask3);   // Geel range
        inRange(hsv, Scalar(85, 90, 70), Scalar(105, 255, 255), mask3);   // Blauw range


        // Combineer
        mask = mask1 | mask2;

        // 3. Ruis verwijderen (optioneel maar sterk aanbevolen)
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        // 3. Ruis verwijderen (optioneel maar sterk aanbevolen)
        morphologyEx(mask3, mask3, MORPH_OPEN, kernel);
        morphologyEx(mask3, mask3, MORPH_CLOSE, kernel);

        // 4. Contours van rood vinden
        vector<vector<Point>> Ballcontours;
        findContours(mask, Ballcontours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<vector<Point>> BorderContours;
        findContours(mask3, BorderContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


        // Check of er überhaupt contours zijn gevonden
        if (Ballcontours.empty()) {

            // Display the image
            imshow("Source", src);
            imshow("Mask", mask);
            imshow("Mask3", mask3);
            
            // ESC = stoppen
            if (waitKey(1) == 27) break;
            continue;
        }

        // Check of er überhaupt contours zijn gevonden
        if (BorderContours.empty()) {

            // Display the image
            imshow("Source", src);
            imshow("Mask", mask);
            imshow("Mask3", mask3);
            
            // ESC = stoppen
            if (waitKey(1) == 27) break;
            continue;
        }


        // 5. Grootste contour pakken (aannemen = bal)
        int largestIndexRed = 0;
        double maxAreaRed = 0;

        // Loop door alle contours en vind de grootste
        for (int i = 0; i < Ballcontours.size(); i++) {

            double areaRed = contourArea(Ballcontours[i]);

            // Alleen overschrijven als de contour groot genoeg is (om ruis te vermijden)
            if (areaRed > maxAreaRed && areaRed > 500) {

                maxAreaRed = areaRed;
                largestIndexRed = i;
            }
        }

        // 5. Grootste contour pakken (aannemen = bal)
        int largestIndexBlue = 0;
        double maxAreaBlue = 0;

        // Loop door alle contours en vind de grootste
        for (int i = 0; i < BorderContours.size(); i++) {

            double areaBlue = contourArea(BorderContours[i]);

            // Alleen overschrijven als de contour groot genoeg is (om ruis te vermijden)
            if (areaBlue > maxAreaBlue && areaBlue > 250) {

                maxAreaBlue = areaBlue;
                largestIndexBlue = i;
            }
        }


        // 6. Middelpunt berekenen (centroid)
        Moments m = moments(Ballcontours[largestIndexRed]);

        int cx = int(m.m10 / m.m00);
        int cy = int(m.m01 / m.m00);

        // Print het middelpunt naar terminal
        cout << "Middelpunt: (" << cx << ", " << cy << ")" << endl;

        // 7. Visualisatie
        circle(src, Point(cx, cy), 5, Scalar(0, 255, 0), -1); // Groen cirkeltje op het middelpunt
        drawContours(src, Ballcontours, largestIndexRed, Scalar(255, 0, 0), 2); // Blauwe contour van de bal
        drawContours(src, BorderContours, largestIndexBlue, Scalar(0, 255, 255), 2); // Gele contour van de border

    

        // Display the image
        imshow("Source", src);
        imshow("HSV", hsv);
        imshow("Mask", mask);
        imshow("Mask3", mask3);

        // ESC = stoppen
        if (waitKey(1) == 27) break;
        
    }

    // Camera vrijgeven
    cap.release();

    // Alle vensters sluiten
    destroyAllWindows();

    return 0;
}