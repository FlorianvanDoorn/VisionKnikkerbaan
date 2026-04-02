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
        inRange(hsv, Scalar(85, 90, 50), Scalar(105, 255, 255), mask3);   // Blauw range


        // Combineer
        mask = mask1 | mask2;

        // 3. Ruis verwijderen (optioneel maar sterk aanbevolen)
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        // 3. Ruis verwijderen (optioneel maar sterk aanbevolen)
        morphologyEx(mask3, mask3, MORPH_OPEN, kernel); // Openen: eerst erode, dan dilate (verwijdert kleine witte ruis)
        morphologyEx(mask3, mask3, MORPH_CLOSE, kernel);    // Sluiten: eerst dilate, dan erode (vult kleine zwarte gaten in de contouren)

        // 4. Contours van rood vinden
        vector<vector<Point>> Ballcontours;   // Vector voor rode contours
        findContours(mask, Ballcontours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);   // Alleen externe contours, geen hiërarchie nodig, eenvoudige benadering

        // 4. Contours van blauw vinden
        vector<vector<Point>> BorderContours;   // Vector voor blauwe contours
        findContours(mask3, BorderContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);    // Alleen externe contours, geen hiërarchie nodig, eenvoudige benadering


        // Check of er überhaupt rode contours zijn gevonden
        if (Ballcontours.empty()) {

            // Display the image
            imshow("Source", src);  // Geef het originele beeld weer.
            imshow("Mask", mask);   // Geef het rode mask weer.
            imshow("Mask3", mask3); // Geef het blauwe mask weer.
            
            // ESC = stoppen
            if (waitKey(1) == 27) break;
            continue;   // Ga terug naar het begin van de loop als er geen rode contours zijn gevonden (om fouten te voorkomen bij het berekenen van het middelpunt)
        }

        // Check of er überhaupt blauwe contours zijn gevonden
        if (BorderContours.empty()) {

            // Display the image
            imshow("Source", src);  // Geef het originele beeld weer.
            imshow("Mask", mask);   // Geef het rode mask weer.
            imshow("Mask3", mask3); // Geef het blauwe mask weer.

            // ESC = stoppen
            if (waitKey(1) == 27) break;
            continue;   // Ga terug naar het begin van de loop als er geen blauwe contours zijn gevonden (om fouten te voorkomen bij het tekenen van de borders)
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
        int largestIndexBlue[2] = {0, 0};
        double maxAreaBlue = 0;

        // Loop door alle contours en vind de grootste
        for (int i = 0; i < BorderContours.size(); i++) {

            double areaBlue = contourArea(BorderContours[i]);

            // Alleen overschrijven als de contour groot genoeg is (om ruis te vermijden)
            if (areaBlue > maxAreaBlue && areaBlue > 500) {

                maxAreaBlue = areaBlue;
                largestIndexBlue[0] = i;
                largestIndexBlue[1] = i-1; // Neem ook de vorige oppervlakte mee (om beide borders te pakken)
            }
        }


        // 6. Middelpunt berekenen (centroid)
        Moments mRed = moments(Ballcontours[largestIndexRed]);

        Moments mBlue1 = moments(BorderContours[largestIndexBlue[0]]);
        Moments mBlue2 = moments(BorderContours[largestIndexBlue[1]]);

        
        int cxRed = int(mRed.m10 / mRed.m00);
        int cyRed = int(mRed.m01 / mRed.m00);

        
        int cxBlue1 = int(mBlue1.m10 / mBlue1.m00);
        int cyBlue1 = int(mBlue1.m01 / mBlue1.m00);

        
        int cxBlue2 = int(mBlue2.m10 / mBlue2.m00);
        int cyBlue2 = int(mBlue2.m01 / mBlue2.m00);
        

        // Print het middelpunt naar terminal
        cout << "Middelpunt: (" << cxRed << ", " << cyRed << ")" << endl;

        cout << "Border 1: (" << cxBlue1 << ", " << cyBlue1 << ")" << endl;
        cout << "Border 2: (" << cxBlue2 << ", " << cyBlue2 << ")" << endl;

        // 7. Visualisatie
        circle(src, Point(cxRed, cyRed), 5, Scalar(0, 255, 0), -1); // Groen cirkeltje op het middelpunt
        circle(src, Point(cxBlue1, cyBlue1), 5, Scalar(0, 0, 0), -1); // Zwart cirkeltje op de border
        circle(src, Point(cxBlue2, cyBlue2), 5, Scalar(0, 0, 0), -1); // Zwart cirkeltje op de border

        
        
        drawContours(src, Ballcontours, largestIndexRed, Scalar(255, 0, 0), 2); // Blauwe contour van de bal
        drawContours(src, BorderContours, largestIndexBlue[0], Scalar(0, 255, 255), 2); // Gele contour van de border
        drawContours(src, BorderContours, largestIndexBlue[1], Scalar(0, 255, 255), 2); // Gele contour van de border

    

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