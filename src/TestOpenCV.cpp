#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>


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

    // Open de seriële poort (vervang "/dev/ttyUSB0" door de juiste poortnaam)
    int serial_port = open("/dev/ttyUSB0", O_RDWR);

    // Check of de seriële poort succesvol is geopend
    if (serial_port < 0) {
        perror("Error opening serial port");
        return -1;
    }

    // Configureer de seriële poort (baudrate, data bits, stop bits, parity, etc.)
    termios tty;    // Struct voor seriële poort instellingen
    tcgetattr(serial_port, &tty);   // Lees huidige instellingen

    // Stel de baudrate in (115200 baud)
    cfsetispeed(&tty, B115200); // Stel de baudrate in (115200 baud)
    cfsetospeed(&tty, B115200); // Stel de baudrate in (115200 baud)

    // Configureer de seriële poort instellingen (8N1, geen flow control)
    tty.c_cflag |= (CLOCAL | CREAD);    // Enable receiver, set local mode
    tty.c_cflag &= ~CSIZE;  // Clear current data bits setting
    tty.c_cflag |= CS8; // Set 8 data bits
    tty.c_cflag &= ~PARENB; // No parity
    tty.c_cflag &= ~CSTOPB; // 1 stop bit
    tty.c_cflag &= ~CRTSCTS;    // No hardware flow control

    // Stel de nieuwe instellingen in
    tcsetattr(serial_port, TCSANOW, &tty);


    while (true) {
        
        // 1. Frame van camera lezen
        cap.read(src);

        // 1. BGR → HSV
        cvtColor(src, hsv, COLOR_BGR2HSV);

        // 2. Mask voor rood (2 ranges!)
        inRange(hsv, Scalar(0, 120, 50), Scalar(10, 255, 255), mask1);  // Lage rood range
        inRange(hsv, Scalar(170, 120, 50), Scalar(180, 255, 255), mask2);   // Hoge rood range

        // inRange(hsv, Scalar(0, 0, 215), Scalar(180, 80, 255), mask3);   // Geel range
        inRange(hsv, Scalar(85, 40, 200), Scalar(99, 255, 255), mask3);   // Blauw range

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
            if (areaBlue > maxAreaBlue && areaBlue > 50) {

                maxAreaBlue = areaBlue;
                largestIndexBlue[0] = i;

                if (i > 0) {
                    largestIndexBlue[1] = i-1; // Neem ook de vorige oppervlakte mee (om beide borders te pakken)
                }
               
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

        


        int dyred = cyRed - cyBlue1;
        int dxred = cxRed - cxBlue1;

        int dyBlue = cyBlue1 - cyBlue2;
        int dxBlue = cxBlue1 - cxBlue2;

        int LengthBlue = sqrt(dxBlue * dxBlue + dyBlue * dyBlue);

        int LengthRed = sqrt(dxred * dxred + dyred * dyred);

        if (LengthBlue < 1) LengthBlue = 1;
        if (LengthRed < 1) LengthRed = 1;

        double ActualPosition = (static_cast<double>(LengthRed) / static_cast<double>(LengthBlue)) * 20.0; // Actuele positie van de bal in cm
        

        // Print het middelpunt naar terminal
        // cout << "Middelpunt: (" << cxRed << ", " << cyRed << ")" << endl;

        // cout << "Border 1: (" << cxBlue1 << ", " << cyBlue1 << ")" << endl;
        // cout << "Border 2: (" << cxBlue2 << ", " << cyBlue2 << ")" << endl;

        // cout << "Afstand tussen borders: " << LengthBlue << " pixels" << endl;
        // cout << "Afstand tussen bal en border: " << LengthRed << " pixels" << endl;
        // cout << "Afstand tussen bal en border: " << ActualPosition << " cm" << endl;

        string msg = "$" + to_string(ActualPosition) + "*\n";
        write(serial_port, msg.c_str(), msg.length());  


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