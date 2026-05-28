// +----------------------------------------------------------------------------------+
// | Project:      Knikkerbaan Regelsysteem                                           |
// | Bestand:      VisionKnikkerbaan.cpp                                              |
// | Auteur:       Florian van Doorn                                                  |
// | Opleiding:    Mechatronica - Avans Hogeschool Breda                              |
// | Datum:        28-05-2026                                                         |
// | Versie:       0.3                                                                |
// |                                                                                  |
// | Beschrijving:                                                                    |
// | Verwerkt camerabeelden met OpenCV om de positie van de knikker                   |
// | te bepalen voor de regelkring.                                                   |
// |                                                                                  |
// | Functionaliteit:                                                                 |
// | - Camera uitlezen                                                                |
// | - HSV filtering                                                                  |
// | - Objectdetectie                                                                 |
// | - Positiebepaling                                                                |
// |                                                                                  |
// | Libraries:                                                                       |
// | - OpenCV                                                                         |
// | - iostream                                                                       | 
// | - algorithm                                                                      |
// | - fcntl.h                                                                        |
// | - unistd.h                                                                       |
// | - termios.h                                                                      |
// +----------------------------------------------------------------------------------+


// used libraries
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>

// namespaces
using namespace std;
using namespace cv;

// global variables
// Images and masks
Mat src;    // Input frame van camera
Mat hsv;    // HSV versie van het frame
Mat mask1;  // Mask voor lage rood range
Mat mask2;  // Mask voor hoge rood range
Mat mask3;  // Mask voor blauw range
Mat mask;   // Gecombineerde mask voor beide rood ranges    

// variabelen voor het berekenen van posities en afstanden
int cxRed = 0;      // X-coördinaat van het middelpunt van de rode contour
int cyRed = 0;      // Y-coördinaat van het middelpunt van de rode contour
int dxRed = 0;      // Verschil in X-coördinaat tussen rode contour en eerste blauwe contour
int dyRed = 0;      // Verschil in Y-coördinaat tussen rode contour en eerste blauwe contour
int LengthRed = 0;  // Afstand tussen rode contour en eerste blauwe contour in pixels

int cxBlue1 = 0;    // X-coördinaat van het middelpunt van de eerste blauwe contour
int cyBlue1 = 0;    // Y-coördinaat van het middelpunt van de eerste blauwe contour
int cxBlue2 = 0;    // X-coördinaat van het middelpunt van de tweede blauwe contour
int cyBlue2 = 0;    // Y-coördinaat van het middelpunt van de tweede blauwe contour
int dxBlue = 0;     // Verschil in X-coördinaat tussen eerste en tweede blauwe contour
int dyBlue = 0;     // Verschil in Y-coördinaat tussen eerste en tweede blauwe contour
int LengthBlue = 0; // Afstand tussen eerste en tweede blauwe contour in pixels

double ActualPosition = 0.0;   // Actuele positie van de knikker in mm (0-200 mm)
double DesiredPosition = 0.0;        // Gewenste positie van de knikker in mm (0-200 mm)

// Trackbar instellingen voor de blauwe border-masker
int blueLowH = 85;
int blueHighH = 99;
int blueHighS = 255;
int blueLowS = 40;
int blueLowV = 200;
int blueHighV = 255;

// Integer trackbars voor double waarden (schaling met 0.1)
int desiredPositionInt = 0;  // 0-2000 (wordt gedeeld door 10 voor 0.0-200.0 mm)

// Proportionele versterking (hoog-res trackbar: schaal 0.001)
int proportionalGainInt = 35; // 1.000 -> waarde = proportionalGainInt / 1000.0
double proportionalGain = 0.035;  // wordt elke iteratie geüpdatet
double DiffValue = 0.6;       // Standaard waarde

void onTrackbar(int, void*) {}

int main() {

    cout << "OpenCV version: " << CV_VERSION << endl;

    // Open default camera
    VideoCapture cap(0, cv::CAP_V4L2); // Gebruik V4L2 backend (Linux) - pas aan indien nodig voor andere platforms
 
    // Check of camera kan worden geopend ...
    if (!cap.isOpened()) {
        cerr << "Can't open camera" << endl;
        return -1;
    }

    // Open de seriële poort
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

    // Maak een regulatiescherm voor de blauwe border-maskers
    namedWindow("ThresholdControls", WINDOW_NORMAL);
    createTrackbar("Blue H min", "ThresholdControls", &blueLowH, 179, onTrackbar);
    createTrackbar("Blue H max", "ThresholdControls", &blueHighH, 179, onTrackbar);
    createTrackbar("Blue S min", "ThresholdControls", &blueLowS, 255, onTrackbar);
    createTrackbar("Blue S max", "ThresholdControls", &blueHighS, 255, onTrackbar);
    createTrackbar("Blue V min", "ThresholdControls", &blueLowV, 255, onTrackbar);
    createTrackbar("Blue V max", "ThresholdControls", &blueHighV, 255, onTrackbar);

    namedWindow("Controls", WINDOW_NORMAL);
    createTrackbar("Desired pos [mm]", "Controls", &desiredPositionInt, 2000, onTrackbar);
    createTrackbar("Kp x0.001", "Controls", &proportionalGainInt, 100000, onTrackbar); // 0.000 - 100.000

    cout << "\n=== Besturingselementen ===" << endl;
    cout << "Gebruik de trackbars: 'Desired pos' en 'Kp x0.001'" << endl;
    cout << "Of: '+' / '-' wijzigingen, of druk 'k' om Kp te sturen, 'P' om desired te pushen" << endl;
    cout << "ESC: Afsluiten" << endl;


    while (true) {
        
        // 1. Frame van camera lezen
        cap.read(src);

        // 1. BGR → HSV
        cvtColor(src, hsv, COLOR_BGR2HSV);

        // 2. Mask voor rood (2 ranges!)
        inRange(hsv, Scalar(0, 120, 50), Scalar(10, 255, 255), mask1);  // Lage rood range
        inRange(hsv, Scalar(170, 120, 50), Scalar(180, 255, 255), mask2);   // Hoge rood range

        // inRange(hsv, Scalar(0, 0, 215), Scalar(180, 80, 255), mask3);   // Geel range
        if (blueLowH > blueHighH) std::swap(blueLowH, blueHighH);
        if (blueLowS > blueHighS) std::swap(blueLowS, blueHighS);
        if (blueLowV > blueHighV) std::swap(blueLowV, blueHighV);
        inRange(hsv, Scalar(blueLowH, blueLowS, blueLowV), Scalar(blueHighH, blueHighS, blueHighV), mask3);   // Blauw range

        // Combineer            
        mask = mask1 | mask2;

        // 3. Ruis verwijderen
        Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5,5));
        morphologyEx(mask, mask, MORPH_OPEN, kernel);
        morphologyEx(mask, mask, MORPH_CLOSE, kernel);

        // 3. Ruis verwijderen
        morphologyEx(mask3, mask3, MORPH_OPEN, kernel); // Openen: eerst erode, dan dilate (verwijdert kleine witte ruis)
        morphologyEx(mask3, mask3, MORPH_CLOSE, kernel);    // Sluiten: eerst dilate, dan erode (vult kleine zwarte gaten in de contouren)

        // 4. Contours van rood vinden
        vector<vector<Point>> Ballcontours;   // Vector voor rode contours
        findContours(mask, Ballcontours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);   // Alleen externe contours, geen hiërarchie nodig, eenvoudige benadering

        // 4. Contours / connected components van blauw vinden
        // Gebruik connectedComponentsWithStats voor robuustere detectie en stabiele middelpuntbepaling
        Mat labels, stats, centroids;
        int nLabels = connectedComponentsWithStats(mask3, labels, stats, centroids);

        // Verzamel componenten (area, centroid)
        vector<pair<int, Point>> comps; // pair<area, center>
        for (int lbl = 1; lbl < nLabels; ++lbl) {
            int area = stats.at<int>(lbl, CC_STAT_AREA);
            if (area > 50) { // filter kleine ruis
                int cx = static_cast<int>(centroids.at<double>(lbl, 0));
                int cy = static_cast<int>(centroids.at<double>(lbl, 1));
                comps.emplace_back(area, Point(cx, cy));
            }
        }

        // Sorteer aflopend op oppervlakte
        sort(comps.begin(), comps.end(), [](const pair<int,Point>& a, const pair<int,Point>& b){ return a.first > b.first; });

        // Maak een eenvoudige lijst met border-centroids (vul met 0 als niet gevonden)
        vector<Point> borderCenters(2, Point(0,0));
        for (size_t i = 0; i < comps.size() && i < 2; ++i) {
            borderCenters[i] = comps[i].second;
        }

        // Teken (optioneel) contours van de gelabelde componenten voor visualisatie
        // (we tekenen de contouren later met drawContours op basis van mask3)


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

        // Check of er überhaupt genoeg blauwe componenten zijn gevonden (minimaal 2 borders nodig)
        if (comps.size() < 2) {
            // Display the image
            imshow("Source", src);  // Geef het originele beeld weer.
            imshow("Mask", mask);   // Geef het rode mask weer.
            imshow("Mask3", mask3); // Geef het blauwe mask weer.

            // ESC = stoppen
            if (waitKey(1) == 27) break;
            continue;   // Ga terug naar het begin van de loop als er niet genoeg borders zijn gevonden
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

        // (oude BorderContours-logica verwijderd; we gebruiken nu connected components)


        // 6. Middelpunt berekenen
        Moments mRed = moments(Ballcontours[largestIndexRed]);          // Bereken het middelpunt van de grootste rode contour (de bal)

        // Bereken de coördinaten van het middelpunt van de rode contour (de bal)
        cxRed = int(mRed.m10 / mRed.m00);
        cyRed = int(mRed.m01 / mRed.m00);

        // Bereken de coördinaten van het middelpunt van de blauwe contouren (de borders)
        if (borderCenters[0].x < borderCenters[1].x) {
            cxBlue1 = borderCenters[0].x;
            cyBlue1 = borderCenters[0].y;
            cxBlue2 = borderCenters[1].x;
            cyBlue2 = borderCenters[1].y;
        } 
        else {    
        cxBlue1 = borderCenters[1].x;
        cyBlue1 = borderCenters[1].y;
        cxBlue2 = borderCenters[0].x;
        cyBlue2 = borderCenters[0].y;
        };


        dyRed = cyRed - cyBlue1;
        dxRed = cxRed - cxBlue1;

        dyBlue = cyBlue1 - cyBlue2;
        dxBlue = cxBlue1 - cxBlue2;

        LengthBlue = sqrt(dxBlue * dxBlue + dyBlue * dyBlue);

        LengthRed = sqrt(dxRed * dxRed + dyRed * dyRed);

        if (LengthBlue < 1) LengthBlue = 1;
        if (LengthRed < 1) LengthRed = 1;

        // Bereken de actuele positie van de bal in mm (0-200 mm)
        ActualPosition = (static_cast<double>(LengthRed) / static_cast<double>(LengthBlue)) * 200.0; // Actuele positie van de bal in mm
        
        // Beperk de waarde van ActualPosition tot het bereik 0-200 mm
        if (ActualPosition < 0.0) ActualPosition = 0.0;
        if (ActualPosition > 200.0) ActualPosition = 200.0;

        // Converteer trackbar integer naar double (0-2000 → 0.0-200.0 mm)
        DesiredPosition = desiredPositionInt / 10.0;

        // Converteer Kp integer (x0.001) naar double (bijv. 1000 -> 1.000)
        proportionalGain = proportionalGainInt / 1000.0;


        // Print het middelpunt naar terminal
        // cout << "Middelpunt: (" << cxRed << ", " << cyRed << ")" << endl;

        // cout << "Border 1: (" << cxBlue1 << ", " << cyBlue1 << ")" << endl;
        // cout << "Border 2: (" << cxBlue2 << ", " << cyBlue2 << ")" << endl;

        // cout << "Afstand tussen borders: " << LengthBlue << " pixels" << endl;
        // cout << "Afstand tussen bal en border: " << LengthRed << " pixels" << endl;
        // cout << "Afstand tussen bal en border: " << ActualPosition << " mm" << endl;

        

        // Stuur de actuele positie van de bal naar de microcontroller via de seriële poort
        string Posmsg = "$Actpos," + to_string(ActualPosition) + "*\n";
        write(serial_port, Posmsg.c_str(), Posmsg.length());

        // 7. Teken het middelpunt van de bal en de uiteinden van de knikkerbaan
        circle(src, Point(cxRed, cyRed), 5, Scalar(0, 255, 0), -1); // Groen cirkeltje op het middelpunt
        circle(src, Point(cxBlue1, cyBlue1), 5, Scalar(0, 0, 0), -1); // Zwart cirkeltje op de border
        circle(src, Point(cxBlue2, cyBlue2), 5, Scalar(0, 0, 0), -1); // Zwart cirkeltje op de border

        
        // Teken de contouren van de bal en de uiteinden van de knikkerbaan
        drawContours(src, Ballcontours, largestIndexRed, Scalar(255, 0, 0), 2); // Blauwe contour van de bal
        // Visualiseer de twee border-centroids en verbind ze
        line(src, borderCenters[0], borderCenters[1], Scalar(0, 255, 255), 2);
        circle(src, borderCenters[0], 10, Scalar(0, 255, 255), 2);
        circle(src, borderCenters[1], 10, Scalar(0, 255, 255), 2);

    

        // geef het beeld weer
        {
            std::ostringstream ss;
            ss << fixed << setprecision(4) << proportionalGain;
            putText(src, "Desired: " + to_string(DesiredPosition) + " mm", Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
            putText(src, string("Kp: ") + ss.str(), Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
            putText(src, "Press P to push, K to send Kp", Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
        }

        imshow("Source", src);
        imshow("HSV", hsv);
        imshow("Mask", mask);
        imshow("Mask3", mask3);

        int key = waitKey(1);
        if (key == '+' || key == '=') {
            // verhoog Kp met 0.001 (alleen update lokaal, geen verzenden)
            proportionalGainInt += 1; // 0.001
            if (proportionalGainInt > 100000) proportionalGainInt = 100000;
            proportionalGain = proportionalGainInt / 1000.0;
            cout << "Kp (updated): " << fixed << setprecision(4) << proportionalGain << " (not sent)" << endl;
        }
        if (key == '-' || key == '_') {
            // verlaag Kp met 0.001 (alleen update lokaal, geen verzenden)
            proportionalGainInt -= 1; // 0.001
            if (proportionalGainInt < 0) proportionalGainInt = 0;
            proportionalGain = proportionalGainInt / 1000.0;
            cout << "Kp (updated): " << fixed << setprecision(4) << proportionalGain << " (not sent)" << endl;
        }
        if (key == 'k' || key == 'K') {
            ostringstream ss; ss << fixed << setprecision(4) << proportionalGain;
            string Kpmsg = "$Kp," + ss.str() + "*\n";
            write(serial_port, Kpmsg.c_str(), Kpmsg.length());
            cout << "Sent Kp: " << ss.str() << endl;
        }
        if (key == 'p' || key == 'P') {
            // stuur desired position en Kp
            ostringstream ss; ss << fixed << setprecision(1) << DesiredPosition;
            string Desirmsg = "$Desirpos," + ss.str() + "*\n";
            write(serial_port, Desirmsg.c_str(), Desirmsg.length());
            ostringstream ss2; ss2 << fixed << setprecision(4) << proportionalGain;
            string Kpmsg = "$Kp," + ss2.str() + "*\n";
            write(serial_port, Kpmsg.c_str(), Kpmsg.length());
            cout << "Pushed desired position: " << ss.str() << " mm and Kp=" << ss2.str() << endl;
        }
        if (key == 27) break;
        
    }

    // Camera vrijgeven
    cap.release();

    // Alle vensters sluiten
    destroyAllWindows();

    return 0;
}